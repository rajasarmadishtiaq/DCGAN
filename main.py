import tensorflow as tf
from tensorflow.keras.layers import Flatten, Reshape, BatchNormalization, Dense, LeakyReLU, Conv2D, Conv2DTranspose, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import imageio
import matplotlib
import PIL
import os
import matplotlib.pyplot as plt
from google.colab import drive
import zipfile
import glob

!mkdir generated_images

with zipfile.ZipFile('/content/drive/MyDrive/GAN Images/archive.zip') as z:
  z.extractall('.')

imgs_path = ('/content/cats')
imgs = glob.glob('/content/cats/*.jpg')
img_width = 96
img_height = 96
channels = 3
img_shape = (img_width, img_height, channels)
latent_dim = 100
learning_rate = 1.5e-4
beta = 0.5
batch_size = 64
rows = 4
cols = 7
margin = 16

datagen = ImageDataGenerator(rescale=1./255, rotation_range = 20, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest')
train_generator = datagen.flow_from_directory(imgs_path, target_size = (img_width, img_height), batch_size = batch_size, class_mode = None)
x = next(train_generator)

print(x.shape)

x_train = next(train_generator)

print(x_train.shape)

plt.imshow(x_train[4])

def save_imgs(count, noise):
  image_array = np.full(( 
      margin + (rows * ((img_width) + margin)), 
      margin + (rows * ((img_width) +margin)), channels), 
      255, dtype=np.uint8)
  
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(rows):
      for col in range(cols):
        r = row * ((img_width)+16) + margin
        c = col * ((img_width)+16) + margin
        image_array[r:r+(img_width),c:c+(img_width)] \
            = generated_images[image_count] * 255
        image_count += 1

          
  output_path = ('/content/generated_images')
  
  filename = os.path.join(output_path,f"train-{count}.png")
  im = Image.fromarray(image_array)
  im.save(filename)


def build_discriminator():

  model = Sequential()

  model.add(Conv2D(32, (3, 3), strides = (2, 2), padding = 'same', input_shape = (img_shape)))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.2))

  model.add(Conv2D(64, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.2))

  model.add(Conv2D(128, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.2))

  model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.2))

  model.add(Conv2D(512, (3, 3), strides = (1, 1), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.2))
  
  #GlobalAveragePooling2D()

  model.add(Flatten())
  
  model.add(Dense(1, activation = 'sigmoid'))

  return model

discriminator = build_discriminator()

#discriminator.compile(optimizer = Adam(learning_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])

upscale_dims = (256, 4, 4)

print(discriminator.summary())

def build_generator():

  model = Sequential()

  model.add(Dense(np.prod(upscale_dims), input_dim = latent_dim))

  model.add(LeakyReLU(alpha = 0.2))

  model.add(Reshape((4, 4, 256)))

  model.add(Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha = 0.2))

  model.add(Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha = 0.2))

  model.add(Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha = 0.2))

  model.add(Conv2DTranspose(128, (3, 3), strides = (3, 3), padding = 'same'))
  model.add(BatchNormalization(momentum = 0.8))
  model.add(LeakyReLU(alpha = 0.2))

  model.add(Conv2DTranspose(3, (3, 3), activation = 'tanh', padding = 'same'))

  return model

generator = build_generator()

print(generator.summary())

generator = build_generator()

discriminator = build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy()

discriminator_optimizer = Adam(learning_rate, beta)
generator_optimizer = Adam(learning_rate, beta)

def discriminator_loss(real, fake):

  real_loss = cross_entropy(tf.ones_like(real), real)
  fake_loss = cross_entropy(tf.zeros_like(fake), fake)
  total_loss = real_loss + fake_loss

  return total_loss

def generator_loss(fake):

  return cross_entropy(tf.ones_like(fake), fake)

@tf.function

def train_step(images):

  noise = tf.random.normal([batch_size, latent_dim])

  with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

    generated_images = generator(noise, training = True)

    real_prediction = discriminator(images, training = True)
    fake_prediction = discriminator(generated_images, training = True)

    gen_loss = generator_loss(fake_prediction)
    disc_loss = discriminator_loss(real_prediction, fake_prediction)

    generator_gradients = generator_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = discriminator_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  return gen_loss, disc_loss

def train(epochs, save_interval):
  
  noise = np.random.normal(0, 1, (rows, cols, latent_dim))

  x_train = next(train_generator)

  for epoch in range(epochs):

    #x_train = next(train_generator) experinment by uncommenting this and commenting the above x_train

    generator_loss_list = []
    discriminator_loss_list = []

    for image_batch in x_train:

      image_batch = np.reshape(image_batch, (1, img_width, img_height, channels))

      t = train_step(image_batch)
      generator_loss_list.append(t[0])
      discriminator_loss_list.append(t[1])

    gen_loss = sum(generator_loss_list) / len(generator_loss_list)
    disc_loss = sum(discriminator_loss_list) / len(discriminator_loss_list)

    print("******* %d [D loss: %f] [G loss: %f]" % (epoch, disc_loss, gen_loss))
    
    if(epoch % save_interval == 0):
      save_imgs(epoch, noise)

train(30000, 10)

anim_file = 'dcgan_cats.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    
    filenames = glob.glob('generated_images/*.png')
    filenames = sorted(filenames)
    
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

  image = imageio.imread(filename)
  writer.append_data(image)
