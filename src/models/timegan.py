import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

def timegan_init(time_series_len, features, rnn_units, rnn_layers):
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
    _, generator, supervisor, recovery, _ = timegan_tuple
    syn_gen_input = keras.Input(generator.input_shape[1:])
    syn_gen_output = generator(syn_gen_input)
    syn_gen_output = supervisor(syn_gen_output)
    syn_gen_output = recovery(syn_gen_output)
    return keras.Model(syn_gen_input, syn_gen_output)
    
def timegan_train(x, timegan_tuple, epochs, batch_size, learning_rate, test_size=0.2):
    # convert to float32 (because random_vector's type is float32, should be matched)
    x = x.astype(np.float32)
    
    # Mezclar los índices de los datos para una partición aleatoria
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * (1 - test_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    x_train = x[train_indices]
    x_test = x[test_indices]
    
    # Verificar las diferencias estadísticas entre train y test antes de empezar
    print("Verificando estadísticas de los conjuntos:")
    print(f"Train - Media: {np.mean(x_train):.4f}, Desv. Est.: {np.std(x_train):.4f}")
    print(f"Test - Media: {np.mean(x_test):.4f}, Desv. Est.: {np.std(x_test):.4f}")
    
    # Funciones para obtener lotes asegurando el mismo tamaño
    def get_batch():
        indices = np.random.permutation(x_train.shape[0])[:batch_size]
        return tf.convert_to_tensor(x_train[indices])
    
    def get_test_batch():
        # Si hay menos muestras de test que el tamaño del lote, muestreamos con reemplazo
        if x_test.shape[0] < batch_size:
            indices = np.random.choice(x_test.shape[0], batch_size, replace=True)
        else:
            indices = np.random.permutation(x_test.shape[0])[:batch_size]
        return tf.convert_to_tensor(x_test[indices])
    
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
            h = embedder(x)
            x_tilde = recovery(h)
            loss = 10 * tf.sqrt(mse(x_tilde, x))
        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def test_autoencoder(x, timegan, mse):
        embedder, generator, supervisor, recovery, discriminator = timegan
        h = embedder(x)
        x_tilde = recovery(h)
        loss = 10 * tf.sqrt(mse(x_tilde, x))
        return loss

    @tf.function
    def train_supervisor(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            h = embedder(x)
            h_pred = supervisor(h)
            loss = mse(h[:, 1:, :], h_pred[:, :-1, :])
        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(loss, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return loss

    @tf.function
    def test_supervisor(x, timegan, mse):
        embedder, generator, supervisor, recovery, discriminator = timegan
        h = embedder(x)
        h_pred = supervisor(h)
        loss = mse(h[:, 1:, :], h_pred[:, :-1, :])
        return loss

    @tf.function
    def train_generator(x, z, timegan, mse, bce, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            # supervised loss
            h = embedder(x)
            h_pred = supervisor(h)
            supervised_loss = mse(h[:, 1:, :], h_pred[:, :-1, :])

            # unsupervised loss
            y_real = tf.ones((x.shape[0], x.shape[1], 1))
            h_fake = generator(z)
            h_fake_sup = supervisor(h_fake)
            y_fake_sup = discriminator(h_fake_sup)
            unsupervised_loss = bce(y_real, y_fake_sup)

            # unsupervised loss - E
            y_real = tf.ones((x.shape[0], x.shape[1], 1))
            h_fake = generator(z)
            y_fake = discriminator(h_fake)
            unsupervised_loss_e = bce(y_real, y_fake)

            # moment loss
            x_real = x
            h_fake = generator(z)
            h_fake_sup = supervisor(h_fake)
            x_fake = recovery(h_fake_sup)
            
            x_real_mean, x_real_var = tf.nn.moments(x_real, axes=[0])
            x_fake_mean, x_fake_var = tf.nn.moments(x_fake, axes=[0])
            
            v1 = tf.reduce_mean(tf.abs(x_real_mean - x_fake_mean))
            v2 = tf.reduce_mean(tf.abs(tf.sqrt(x_real_var + 1e-6) - tf.sqrt(x_fake_var + 1e-6)))
            moment_loss = v1 + v2

            loss = supervised_loss + 100 * tf.sqrt(unsupervised_loss) + unsupervised_loss_e + 100 * moment_loss

        var_list = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss, supervised_loss, unsupervised_loss, moment_loss

    @tf.function
    def test_generator(x, z, timegan, mse, bce):
        embedder, generator, supervisor, recovery, discriminator = timegan
        # supervised loss
        h = embedder(x)
        h_pred = supervisor(h)
        supervised_loss = mse(h[:, 1:, :], h_pred[:, :-1, :])

        # unsupervised loss
        y_real = tf.ones((x.shape[0], x.shape[1], 1))
        h_fake = generator(z)
        h_fake_sup = supervisor(h_fake)
        y_fake_sup = discriminator(h_fake_sup)
        unsupervised_loss = bce(y_real, y_fake_sup)

        # unsupervised loss - E
        y_real = tf.ones((x.shape[0], x.shape[1], 1))
        h_fake = generator(z)
        y_fake = discriminator(h_fake)
        unsupervised_loss_e = bce(y_real, y_fake)

        # moment loss
        x_real = x
        h_fake = generator(z)
        h_fake_sup = supervisor(h_fake)
        x_fake = recovery(h_fake_sup)
        
        x_real_mean, x_real_var = tf.nn.moments(x_real, axes=[0])
        x_fake_mean, x_fake_var = tf.nn.moments(x_fake, axes=[0])
        
        v1 = tf.reduce_mean(tf.abs(x_real_mean - x_fake_mean))
        v2 = tf.reduce_mean(tf.abs(tf.sqrt(x_real_var + 1e-6) - tf.sqrt(x_fake_var + 1e-6)))
        moment_loss = v1 + v2

        loss = supervised_loss + 100 * tf.sqrt(unsupervised_loss) + unsupervised_loss_e + 100 * moment_loss
        return loss, supervised_loss, unsupervised_loss, moment_loss

    @tf.function
    def train_embedder(x, timegan, mse, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            # supervised loss
            h = embedder(x)
            h_pred = supervisor(h)
            supervised_loss = mse(h[:, 1:, :], h_pred[:, :-1, :])

            # reconstruction loss
            h = embedder(x)
            x_tilde = recovery(h)
            reconstruction_loss = 10 * tf.sqrt(mse(x_tilde, x))

            loss = reconstruction_loss + 0.1 * supervised_loss

        var_list = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss, reconstruction_loss, supervised_loss

    @tf.function
    def test_embedder(x, timegan, mse):
        embedder, generator, supervisor, recovery, discriminator = timegan
        # supervised loss
        h = embedder(x)
        h_pred = supervisor(h)
        supervised_loss = mse(h[:, 1:, :], h_pred[:, :-1, :])

        # reconstruction loss
        h = embedder(x)
        x_tilde = recovery(h)
        reconstruction_loss = 10 * tf.sqrt(mse(x_tilde, x))

        loss = reconstruction_loss + 0.1 * supervised_loss
        return loss, reconstruction_loss, supervised_loss

    @tf.function
    def train_discriminator(x, z, timegan, bce, opt):
        embedder, generator, supervisor, recovery, discriminator = timegan
        with tf.GradientTape() as tape:
            # loss on FN
            y_real = tf.ones((x.shape[0], x.shape[1], 1))
            h = embedder(x)
            y_pred_real = discriminator(h)
            loss_real = bce(y_real, y_pred_real)

            # loss on FP
            y_fake = tf.zeros((x.shape[0], x.shape[1], 1))
            h_fake = generator(z)
            h_fake_sup = supervisor(h_fake)
            y_pred_fake_sup = discriminator(h_fake_sup)
            loss_fake_sup = bce(y_fake, y_pred_fake_sup)

            # loss on FP - E
            y_fake = tf.zeros((x.shape[0], x.shape[1], 1))
            h_fake = generator(z)
            y_pred_fake = discriminator(h_fake)
            loss_fake = bce(y_fake, y_pred_fake)           

            loss = loss_real + loss_fake_sup + loss_fake

        var_list = discriminator.trainable_variables
        gradients = tape.gradient(loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return loss

    @tf.function
    def test_discriminator(x, z, timegan, bce):
        embedder, generator, supervisor, recovery, discriminator = timegan
        # loss on FN
        y_real = tf.ones((x.shape[0], x.shape[1], 1))
        h = embedder(x)
        y_pred_real = discriminator(h)
        loss_real = bce(y_real, y_pred_real)

        # loss on FP
        y_fake = tf.zeros((x.shape[0], x.shape[1], 1))
        h_fake = generator(z)
        h_fake_sup = supervisor(h_fake)
        y_pred_fake_sup = discriminator(h_fake_sup)
        loss_fake_sup = bce(y_fake, y_pred_fake_sup)

        # loss on FP - E
        y_fake = tf.zeros((x.shape[0], x.shape[1], 1))
        h_fake = generator(z)
        y_pred_fake = discriminator(h_fake)
        loss_fake = bce(y_fake, y_pred_fake)           

        loss = loss_real + loss_fake_sup + loss_fake
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
        
        # Asegúrate de que ambos lotes tienen el mismo tamaño
        assert batch.shape[0] == test_batch.shape[0], "Los lotes de train y test deben tener el mismo tamaño"
        
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
            
            train_loss_tuple = train_generator(batch, random_vector, timegan_tuple, mse, bce, opt_generator)
            test_loss_tuple = test_generator(test_batch, random_vector, timegan_tuple, mse, bce)
            
            gen_train_loss += train_loss_tuple[0].numpy()
            gen_test_loss += test_loss_tuple[0].numpy()
            
            emb_train_loss_tuple = train_embedder(batch, timegan_tuple, mse, opt_embedder)
            emb_test_loss_tuple = test_embedder(test_batch, timegan_tuple, mse)
            
            emb_train_loss += emb_train_loss_tuple[0].numpy()
            emb_test_loss += emb_test_loss_tuple[0].numpy()

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
    plt.savefig('../images/timegan_training_loss.png')
    plt.show()
    
    # Al finalizar, imprimir información sobre la generación
    print("\nPropiedades estadísticas después del entrenamiento:")
    
    # Generar algunas secuencias para comparar
    random_vector = get_random_vector()
    h_fake = timegan_tuple[1](random_vector)  # generator
    h_fake_sup = timegan_tuple[2](h_fake)     # supervisor
    x_fake = timegan_tuple[3](h_fake_sup)     # recovery
    
    batch = get_batch()
    
    print(f"Datos reales - Media: {np.mean(batch):.4f}, Desv. Est.: {np.std(batch):.4f}")
    print(f"Datos generados - Media: {np.mean(x_fake):.4f}, Desv. Est.: {np.std(x_fake):.4f}")

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