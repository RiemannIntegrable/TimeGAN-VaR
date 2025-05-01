import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

def timegan_init(time_series_len, features, rnn_units, rnn_layers):
    # El código existente permanece igual
    def get_model(input_shape, output_units, rnn_units, layer_cnt):
        inputs = keras.layers.Input(input_shape)
        x = inputs
        for _ in range(layer_cnt):
            x = keras.layers.GRU(rnn_units, return_sequences=True)(x)
        outputs = keras.layers.Dense(output_units, activation="sigmoid")(x)
        return keras.Model(inputs, outputs)

    input_shape = (time_series_len, features)
    latent_code_shape = (time_series_len, rnn_units)
    embedder = get_model(input_shape, rnn_units, rnn_units, rnn_layers)
    generator = get_model(input_shape, rnn_units, rnn_units, rnn_layers)
    supervisor = get_model(latent_code_shape, rnn_units, rnn_units, rnn_layers)
    recovery = get_model(latent_code_shape, features, rnn_units, rnn_layers)
    discriminator = get_model(latent_code_shape, 1, rnn_units, rnn_layers-1)
    return embedder, generator, supervisor, recovery, discriminator

def timegan_export_generator(timegan_tuple):
    # El código existente permanece igual
    _, generator, supervisor, recovery, _ = timegan_tuple
    syn_gen_input = keras.Input(generator.input_shape[1:])
    syn_gen_output = generator(syn_gen_input)
    syn_gen_output = supervisor(syn_gen_output)
    syn_gen_output = recovery(syn_gen_output)
    return keras.Model(syn_gen_input, syn_gen_output)
    
def timegan_train(x, timegan_tuple, epochs, batch_size, learning_rate, test_size=0.2):
    # convert to float32 (because random_vector's type is float32, should be matched)
    x = x.astype(np.float32)
    
    # Dividir los datos en train y test respetando el orden cronológico
    n_samples = x.shape[0]
    train_size = int(n_samples * (1 - test_size))
    
    # Los primeros train_size elementos son para entrenamiento
    x_train = x[:train_size]
    # Los últimos n_samples - train_size elementos son para prueba
    x_test = x[train_size:]
    
    # Funciones para obtener lotes
    get_batch = lambda: tf.convert_to_tensor(x_train[np.random.permutation(x_train.shape[0])[:batch_size]])
    get_test_batch = lambda: tf.convert_to_tensor(x_test[np.random.permutation(x_test.shape[0])[:min(batch_size, x_test.shape[0])]])
    get_random_vector = lambda: tf.convert_to_tensor(np.random.uniform(size=(batch_size, x.shape[1], x.shape[2])))

    # loss & optimizer
    mse = keras.losses.MeanSquaredError()
    bce = keras.losses.BinaryCrossentropy()
    opt_autoencoder = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_supervisor = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_generator = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_embedder = keras.optimizers.Adam(learning_rate=learning_rate)
    opt_discriminator = keras.optimizers.Adam(learning_rate=learning_rate)

    # training functions
    @tf.function
    def train_autoencoder(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            y_true = embedder(x)
            y_true = recovery(y_true)
            loss = 10 * tf.sqrt(mse(y_true, x))
        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def test_autoencoder(x, timegan, mse):
        embedder, generator, supervisor, recovery, discriminator = timegan
        y_true = embedder(x)
        y_true = recovery(y_true)
        loss = 10 * tf.sqrt(mse(y_true, x))
        return loss

    @tf.function
    def train_supervisor(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            y_true = embedder(x)
            y_pred = supervisor(y_true)
            loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])
        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(loss, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return loss

    @tf.function
    def test_supervisor(x, timegan, mse):
        embedder, generator, supervisor, recovery, discriminator = timegan
        y_true = embedder(x)
        y_pred = supervisor(y_true)
        loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])
        return loss

    @tf.function
    def train_generator(x, z, timegan, mse, bce, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            # supervised loss
            y_true = embedder(x)
            y_pred = supervisor(y_true)
            supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

            # unsupervised loss
            y_true = tf.ones((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = supervisor(y_pred)
            y_pred = discriminator(y_pred)
            unsupervised_loss = bce(y_true, y_pred)

            # unsupervised loss - E
            y_true = tf.ones((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = discriminator(y_pred)
            unsupervised_loss_e = bce(y_true, y_pred)

            # moment loss
            y_true = x
            y_pred = generator(z)
            y_pred = supervisor(y_pred)
            y_pred = recovery(y_pred)
            y_true_mean, y_true_var = tf.nn.moments(y_true, axes=[0])
            y_pred_mean, y_pred_var = tf.nn.moments(y_pred, axes=[0])
            v1 = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
            v2 = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
            moment_loss = v1 + v2

            loss = supervised_loss + 100 * tf.sqrt(unsupervised_loss) + unsupervised_loss_e + 100 * moment_loss

        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def test_generator(x, z, timegan, mse, bce):
        embedder, generator, supervisor, recovery, discriminator = timegan
        # supervised loss
        y_true = embedder(x)
        y_pred = supervisor(y_true)
        supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

        # unsupervised loss
        y_true = tf.ones((x.shape[0], x.shape[1], 1))
        y_pred = generator(z)
        y_pred = supervisor(y_pred)
        y_pred = discriminator(y_pred)
        unsupervised_loss = bce(y_true, y_pred)

        # unsupervised loss - E
        y_true = tf.ones((x.shape[0], x.shape[1], 1))
        y_pred = generator(z)
        y_pred = discriminator(y_pred)
        unsupervised_loss_e = bce(y_true, y_pred)

        # moment loss
        y_true = x
        y_pred = generator(z)
        y_pred = supervisor(y_pred)
        y_pred = recovery(y_pred)
        y_true_mean, y_true_var = tf.nn.moments(y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(y_pred, axes=[0])
        v1 = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        v2 = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
        moment_loss = v1 + v2

        loss = supervised_loss + 100 * tf.sqrt(unsupervised_loss) + unsupervised_loss_e + 100 * moment_loss
        return loss

    @tf.function
    def train_embedder(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            # supervised loss
            y_true = embedder(x)
            y_pred = supervisor(y_true)
            supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

            # reconstruction loss
            y_true = embedder(x)
            y_true = recovery(y_true)
            y_pred = x
            reconstruction_loss = 10 * tf.sqrt(mse(y_true, y_pred))

            loss = reconstruction_loss + 0.1 * supervised_loss

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def test_embedder(x, timegan, mse):
        embedder, generator, supervisor, recovery, discriminator = timegan
        # supervised loss
        y_true = embedder(x)
        y_pred = supervisor(y_true)
        supervised_loss = mse(y_true[:, 1:, :], y_pred[:, :-1, :])

        # reconstruction loss
        y_true = embedder(x)
        y_true = recovery(y_true)
        y_pred = x
        reconstruction_loss = 10 * tf.sqrt(mse(y_true, y_pred))

        loss = reconstruction_loss + 0.1 * supervised_loss
        return loss

    @tf.function
    def train_discriminator(x, z, timegan, bce, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            # loss on FN
            y_true = tf.ones((x.shape[0], x.shape[1], 1))
            y_pred = embedder(x)
            y_pred = discriminator(y_pred)
            loss_on_FN = bce(y_true, y_pred)

            # loss on FP
            y_true = tf.zeros((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = supervisor(y_pred)
            y_pred = discriminator(y_pred)
            loss_on_FP = bce(y_true, y_pred)

            # loss on FP - E
            y_true = tf.zeros((x.shape[0], x.shape[1], 1))
            y_pred = generator(z)
            y_pred = discriminator(y_pred)
            loss_on_FP_E = bce(y_true, y_pred)           

            loss = loss_on_FN + loss_on_FP + loss_on_FP_E

        var_list = discriminator.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def test_discriminator(x, z, timegan, bce):
        embedder, generator, supervisor, recovery, discriminator = timegan
        # loss on FN
        y_true = tf.ones((x.shape[0], x.shape[1], 1))
        y_pred = embedder(x)
        y_pred = discriminator(y_pred)
        loss_on_FN = bce(y_true, y_pred)

        # loss on FP
        y_true = tf.zeros((x.shape[0], x.shape[1], 1))
        y_pred = generator(z)
        y_pred = supervisor(y_pred)
        y_pred = discriminator(y_pred)
        loss_on_FP = bce(y_true, y_pred)

        # loss on FP - E
        y_true = tf.zeros((x.shape[0], x.shape[1], 1))
        y_pred = generator(z)
        y_pred = discriminator(y_pred)
        loss_on_FP_E = bce(y_true, y_pred)           

        loss = loss_on_FN + loss_on_FP + loss_on_FP_E
        return loss

    # Inicializar historiales de pérdida
    autoencoder_train_losses = []
    autoencoder_test_losses = []
    supervisor_train_losses = []
    supervisor_test_losses = []
    generator_train_losses = []
    generator_test_losses = []
    discriminator_train_losses = []
    discriminator_test_losses = []
    
    # conduct train
    print(f"train autoencoder")
    for epoch in trange(epochs):
        batch = get_batch()
        test_batch = get_test_batch()
        
        train_loss = train_autoencoder(batch, timegan_tuple, mse, opt_autoencoder)
        test_loss = test_autoencoder(test_batch, timegan_tuple, mse)
        
        autoencoder_train_losses.append(train_loss.numpy())
        autoencoder_test_losses.append(test_loss.numpy())

    print(f"train supervisor")
    for epoch in trange(epochs):
        batch = get_batch()
        test_batch = get_test_batch()
        
        train_loss = train_supervisor(batch, timegan_tuple, mse, opt_supervisor)
        test_loss = test_supervisor(test_batch, timegan_tuple, mse)
        
        supervisor_train_losses.append(train_loss.numpy())
        supervisor_test_losses.append(test_loss.numpy())

    print(f"joint train")
    for epoch in trange(epochs):
        # Training generator and embedder
        gen_train_loss = 0
        gen_test_loss = 0
        emb_train_loss = 0
        emb_test_loss = 0
        
        for _ in range(2):
            batch = get_batch()
            test_batch = get_test_batch()
            random_vector = get_random_vector()
            
            train_batch_gen_loss = train_generator(batch, random_vector, timegan_tuple, mse, bce, opt_generator)
            test_batch_gen_loss = test_generator(test_batch, random_vector, timegan_tuple, mse, bce)
            gen_train_loss += train_batch_gen_loss.numpy()
            gen_test_loss += test_batch_gen_loss.numpy()
            
            train_batch_emb_loss = train_embedder(batch, timegan_tuple, mse, opt_embedder)
            test_batch_emb_loss = test_embedder(test_batch, timegan_tuple, mse)
            emb_train_loss += train_batch_emb_loss.numpy()
            emb_test_loss += test_batch_emb_loss.numpy()

        # Training discriminator
        batch = get_batch()
        test_batch = get_test_batch()
        random_vector = get_random_vector()
        
        disc_train_loss = train_discriminator(batch, random_vector, timegan_tuple, bce, opt_discriminator)
        disc_test_loss = test_discriminator(test_batch, random_vector, timegan_tuple, bce)
        
        # Record losses
        generator_train_losses.append(gen_train_loss / 2)
        generator_test_losses.append(gen_test_loss / 2)
        discriminator_train_losses.append(disc_train_loss.numpy())
        discriminator_test_losses.append(disc_test_loss.numpy())

    # Graficar las pérdidas después del entrenamiento
    plt.figure(figsize=(16, 12))
    
    # Autoencoder loss
    plt.subplot(2, 2, 1)
    plt.plot(range(len(autoencoder_train_losses)), autoencoder_train_losses, label='Train', color='#1f77b4')
    plt.plot(range(len(autoencoder_test_losses)), autoencoder_test_losses, label='Test', color='#ff7f0e')
    plt.title('Autoencoder Loss', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Supervisor loss
    plt.subplot(2, 2, 2)
    plt.plot(range(len(supervisor_train_losses)), supervisor_train_losses, label='Train', color='#1f77b4')
    plt.plot(range(len(supervisor_test_losses)), supervisor_test_losses, label='Test', color='#ff7f0e')
    plt.title('Supervisor Loss', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Generator loss
    plt.subplot(2, 2, 3)
    plt.plot(range(len(generator_train_losses)), generator_train_losses, label='Train', color='#1f77b4')
    plt.plot(range(len(generator_test_losses)), generator_test_losses, label='Test', color='#ff7f0e')
    plt.title('Generator Loss', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Discriminator loss
    plt.subplot(2, 2, 4)
    plt.plot(range(len(discriminator_train_losses)), discriminator_train_losses, label='Train', color='#1f77b4')
    plt.plot(range(len(discriminator_test_losses)), discriminator_test_losses, label='Test', color='#ff7f0e')
    plt.title('Discriminator Loss', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timegan_training_loss.png')
    plt.show()

    return timegan_tuple

def generator_save(syn_gen, save_path):
    syn_gen.save_weights(save_path)

def generator_load(save_path, time_series_len, features, rnn_units, rnn_layers):
    timegan = timegan_init(time_series_len, features, rnn_units, rnn_layers)
    syn_gen = timegan_export_generator(timegan)
    syn_gen.load_weights(save_path)
    return syn_gen

def generator_gen(syn_gen, generate_cnt):
    syn = syn_gen.predict(np.random.uniform(size=(generate_cnt, syn_gen.input_shape[1], syn_gen.input_shape[2])))
    return syn