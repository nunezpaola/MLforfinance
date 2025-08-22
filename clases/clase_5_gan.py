"""Clase 5: Red Generativa Adversarial (GAN)"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "torch"

import keras as k
from tqdm import tqdm


def create_generator(hu=32):
    model = k.models.Sequential([
        k.layers.Dense(hu, activation="relu"),
        k.layers.Dense(hu, activation="relu"),
        k.layers.Dense(1, activation="linear")
    ])
    return model


def create_discriminator(hu=32):
    model = k.models.Sequential([
        k.layers.Dense(hu, activation="relu"),
        k.layers.Dense(hu, activation="relu"),
        k.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=k.optimizers.Adam())
    return model


def train_gan(data, generator, discriminator, epochs, batch_size, lr=0.001):
    d_losses = []
    g_losses = []

    y_tensor = k.ops.convert_to_tensor(data, dtype="float32")
    real_labels = k.ops.ones((batch_size,), dtype="float32")
    fake_labels = k.ops.zeros((batch_size,), dtype="float32")

    gan = k.models.Sequential([
        generator,
        discriminator
    ])
    gan.compile(loss="binary_crossentropy", optimizer=k.optimizers.Adam(learning_rate=lr))

    for _ in (epoch_bar := tqdm(range(epochs), desc="Training Models", postfix={"D Loss": "?", "G Loss": "?"})):
        # --- Train discriminator ---
        # Real samples
        random_indices = k.random.randint((batch_size,), 0, y_tensor.shape[0])
        real_data = k.ops.take(y_tensor, random_indices, axis=0)

        # Generate synthetic data
        noise = k.random.normal((batch_size, 1))
        synthetic_data = generator(noise)

        # Train discriminator on both real and fake data
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(synthetic_data, fake_labels)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # --- Train generator (through GAN model) ---
        noise = k.random.normal((batch_size, 1))
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, real_labels)  # Try to fool discriminator
        discriminator.trainable = True

        # --- Log losses ---
        # Store losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        # Update progress bar
        epoch_bar.set_postfix({"D Loss": f"{d_loss:.4f}", "G Loss": f"{g_loss:.4f}"})

    return d_losses, g_losses


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import sklearn.preprocessing as sk_pp
    import matplotlib.pyplot as plt

    # Load and preprocess data
    raw_data = pd.read_csv("./data/spx.csv", index_col=0, parse_dates=True)
    rets = raw_data.loc["2018":].pct_change().dropna().values

    scaler = sk_pp.StandardScaler()
    y = scaler.fit_transform(rets)

    # Create models
    generator = create_generator()
    discriminator = create_discriminator()

    # Train models
    d_losses, g_losses = train_gan(y, generator, discriminator, 5000, 50)

    # Generate synth data
    noise = np.random.normal(size=(len(y), 1))
    synth_data_scaled = generator.predict(noise, verbose=0)
    synth_data = scaler.inverse_transform(synth_data_scaled)  # Inverse transform to original scale

    # Plot results
    plt.style.use("dark_background")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    axs[0][0].set_title("Data")
    axs[0][0].plot(rets, label="Real", color="orange")
    axs[0][0].plot(synth_data, label="Synth", color="white")
    axs[0][0].legend()
    axs[0][0].grid(alpha=0.2)

    axs[0][1].set_title("Training Losses")
    axs[0][1].plot(d_losses, label="Discriminator Loss", color="red")
    axs[0][1].plot(g_losses, label="Generator Loss", color="blue")
    axs[0][1].legend()
    axs[0][1].grid(alpha=0.2)

    axs[1][0].set_title("CDF")
    axs[1][0].plot(np.sort(rets, axis=0), lw=1.0, label="Real", color="orange")
    axs[1][0].plot(np.sort(synth_data, axis=0), lw=1.0, label="Synth", color="white", linestyle="--")
    axs[1][0].legend()
    axs[1][0].grid(alpha=0.2)

    axs[1][1].set_title("Histogram")
    axs[1][1].hist(rets, bins=50, alpha=0.7, label="Real", color="orange")
    axs[1][1].hist(synth_data, bins=50, alpha=0.7, label="Synth", color="white")
    axs[1][1].legend()
    axs[1][1].grid(alpha=0.2)

    fig.tight_layout()
    plt.show()
