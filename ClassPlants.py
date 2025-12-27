import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os.path
from tkinter import filedialog
from PIL import Image
import json
import shutil
from datetime import datetime
import time
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path

class Plants:
    def __init__(self, retrain: bool):
        self.model_name = "Model.keras"
        self.backup_folder = "model_backups"
        self.base_flowers = 'Plants/flowers'
        self.add_flowers = 'Plants/add_flowers'
        self.data_dir = 'Plants/'
        self.batch_size = 32
        self.img_height = 224
        self.img_width = 224
        self.equals_line = "=" * 50

        # Create folder for test data
        self.test_data_dir = 'Plants/test_data'
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Create folder for backups
        os.makedirs(self.backup_folder, exist_ok=True)

        #To work with CPU, because I have old AMD GPU. Tensorflow doesn't support it
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        #Augmentation to increase training sets
        self.data_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.3),
            tf.keras.layers.RandomZoom(0.3),
            tf.keras.layers.RandomContrast(0.3),
            tf.keras.layers.RandomBrightness(0.3),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)])

        if retrain or (not os.path.exists(self.model_name)):
            #Training the first model from zero
            self.model = self.train()
        else:
            #Get a ready-made model
            self.model = tf.keras.models.load_model(self.model_name)
            self.model.summary()
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)

    #Recognize chosen image from PC directory
    def __recognize(self, tech:bool, show_samples:bool):

        image_path = filedialog.askopenfilename(title="Choose a plant")
        if not image_path:
            print("File hasn't chosen")
            return None

        # Check model
        if not hasattr(self, 'class_names') or not self.class_names:
            print("❌ Error: The model class hasn't loaded")
            return None

        if not hasattr(self, 'model') or self.model is None:
            print("❌ Error: The model hasn't loaded!")
            return None

        # Check classes count
        if hasattr(self.model.layers[-1], 'units'):
            model_output_size = self.model.layers[-1].units
            if model_output_size != len(self.class_names):
                print(f"❌ Error: Size discrepancy!")
                print(f"   Model expected: {model_output_size} classes")
                print(f"   In class_names: {len(self.class_names)} classes")
                print(f"   First 10 classes: {self.class_names[:10]}")

                # Correction
                if model_output_size > len(self.class_names):
                    print("⚠️  Add missing classes")
                    for i in range(len(self.class_names), model_output_size):
                        self.class_names.append(f"unknown_{i}")
                    with open('class_names.json', 'w') as f:
                        json.dump(self.class_names, f)
                else:
                    print("⚠️ Cut unnecessary classes")
                    self.class_names = self.class_names[:model_output_size]
                    with open('class_names.json', 'w') as f:
                        json.dump(self.class_names, f)

        img = self.__preprocess_image(image_path, tech)

        if tech:
            print(f"File path: {image_path}")
            print(f"Model input form: {img.shape}")
            print(f"Min and max input tensor: {img.min()}, {img.max()}")

        predictions = self.model.predict(img, verbose=0)
        if tech:
            print(f"Raw predictions: {predictions}")

        predictions_sum = np.sum(predictions[0])
        # softmax layer analys
        output_layer = self.model.layers[-1]
        has_softmax = (hasattr(output_layer, 'activation') and
                       output_layer.activation.__name__ == 'softmax')

        print(f"Output layer has softmax: {has_softmax}")

        if abs(predictions_sum - 1.0) < 0.01 and has_softmax:
            print("✅ Softmax has already in model")
            probabilities = predictions[0]
        elif abs(predictions_sum - 1.0) < 0.01:
            print("⚠️ = ~1.0, but output layer doesn't have softmax")
            probabilities = predictions[0]
        else:
            print("⚠️ Apply softmax to logits")
            probabilities = tf.nn.softmax(predictions[0]).numpy()

        predictions_percent = [round(x * 100, 2) for x in probabilities]

        sorted_indices = sorted(range(len(predictions_percent)),
                                key=lambda x: predictions_percent[x],
                                reverse=True)

        print("\n" + self.equals_line)
        print("Results of recognition")
        print(self.equals_line)

        print(f"\n🏆 Top 5:")
        for i, idx in enumerate(sorted_indices[:5]):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}."

            if idx < len(self.class_names):
                class_name = self.class_names[idx]
            else:
                class_name = f"unknown_{idx}"

            print(f"   {emoji} {class_name:<20} → {predictions_percent[idx]:>6.2f}%")

        #Model confidence analysis
        top_prob = predictions_percent[sorted_indices[0]]
        second_prob = predictions_percent[sorted_indices[1]]
        confidence_gap = top_prob - second_prob

        print("\nConfidence analysis")
        print(f"Difference with second place: {confidence_gap:.2f}%")

        if top_prob < 50:
            print("⚠️ The model is not confident (probability < 50%)")
        elif confidence_gap < 15:
            print("⚠️ Little difference with other classes")
        elif confidence_gap < 5:
            print("❗ Model is critical unsure")
        else:
            print("✅ The model is confident")

        top_class = self.class_names[sorted_indices[0]]

        print(f"\n🔍 Precision details '{top_class}':")
        print(f"Index in model: {sorted_indices[0]}")
        print(f"Confidence: {top_prob:.2f}%")

        # Check class exist in data
        available_classes = self.__get_all_available_classes()
        if top_class not in available_classes:
            print(f"⚠️ Class '{top_class}' hasn't found!")
            print(f"Available classes in data: {len(available_classes)}")

            # Search similar
            similar = [cls for cls in available_classes
                       if top_class.lower() in cls.lower() or cls.lower() in top_class.lower()]
            if similar:
                print(f"Similar classes: {similar}")

        #Show examples of top class
        if show_samples:
            print("Searching examples")

            samples = self.__get_sample_images_for_class(top_class, num_samples=9)

            if samples:
                self.__show_prediction_with_samples(top_class, top_prob, samples)

                print(f"\n👁️  Preview of recognized image:")#To check visual
                original_img = Image.open(image_path).convert('RGB')

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                ax1.imshow(original_img)
                ax1.set_title('Original', fontsize=12, fontweight='bold')
                ax1.axis('off')

                processed_display = (img[0] / 255.0).clip(0, 1)
                ax2.imshow(processed_display)
                ax2.set_title('After processing', fontsize=12, fontweight='bold')
                ax2.axis('off')

                plt.suptitle(f'Probability: {top_prob:.2f}%', fontsize=14)
                plt.tight_layout()
                plt.show()
            else:
                print("⚠️ Can't find examples")

            return top_class

        return top_class

    #Resize and convert image to standard, show result
    def __preprocess_image(self, image_path, tech:bool):
        size = (self.img_height,self.img_width)
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        if tech:
            img.show()
        img_array = np.array(img).astype(np.float32)
        return np.expand_dims(img_array, axis=0)

    def train(self):
        start_time = time.time()
        print("Start training the model. Set options")

        # Create temporary directories for train/test
        temp_base = 'Plants/temp_split'
        train_dir = os.path.join(temp_base, 'train')
        test_dir = os.path.join(temp_base, 'test')

        # Split data train/val/test
        split_result = self.__split_data_into_folders(
            self.base_flowers, train_dir, test_dir, test_size=0.1, val_size=0.1
        )

        if split_result[0] is None:
            print("❌ Failed to split data")
            return None

        train_temp_dir, val_temp_dir, test_temp_dir = split_result

        # Save path to test data
        self.original_test_dir = test_temp_dir

        print(f"📊 Создание датасетов...")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_temp_dir,
            validation_split=0,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        self.class_names = train_ds.class_names
        print(f"📋 Classes: {self.class_names}")
        num_classes = len(self.class_names)

        # Show examples of training images
        print("\n👁️  Examples of training plants:")
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(min(9, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        plt.tight_layout()
        plt.show()

        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_temp_dir,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        # Use pre-trained model for boost speed
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3),
            pooling='avg'
        )

        base_model.trainable = False

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3)),
            tf.keras.layers.Rescaling(1. / 255),
            base_model,
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),#Normalization
            tf.keras.layers.Dropout(0.5),#Prevent overfitting
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),#decrease the value for finer tuning
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')#logits to probability
        ])

        model.summary()# show model structure

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']# goal to accuracy
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        print("🚀 Start training")
        training_start = time.time()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - training_start

        # Fine tuning
        if history.history['val_accuracy'][-1] < 0.85:
            print("\n📈 Start fine tuning")

            base_model.trainable = True
            for layer in base_model.layers[:50]:
                layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )

            history_fine = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=15,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=len(history.epoch)
            )

            # Concatenate the history of learning
            for key in history.history:
                history.history[key].extend(history_fine.history[key][1:])

        # Evaluating on test data
        print("\n📊 Evaluating on test data...")
        if os.path.exists(self.original_test_dir):
            test_ds = tf.keras.utils.image_dataset_from_directory(
                self.original_test_dir,
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                shuffle=False
            )

            test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
            print(f"✅ Test accuracy: {test_accuracy:.4f}")
            print(f"📊 Test loss: {test_loss:.4f}")
        else:
            print("⚠️ Test directory not found")
            test_accuracy = 0.0
            test_loss = 0.0

        model.save(self.model_name)
        print(f"💾 Model saved to {self.model_name}")

        with open('class_names.json', 'w') as f:
            json.dump(self.class_names, f)

        # Clear temporary directory
        try:
            shutil.rmtree(temp_base, ignore_errors=True)
        except Exception as e:
            print(f"⚠️ Clear temporary directory failed: {e}")

        total_time = time.time() - start_time

        print("\n" + self.equals_line)
        print("🎯 TRAINING COMPLETED")
        print(self.equals_line)
        print(f"📊 Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"🧪 Test accuracy: {test_accuracy:.4f}")
        print(f"⏱️  Training time: {training_time / 60:.1f} minutes")
        print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
        print(self.equals_line)

        self.__plot_training_history(history)

        return model

    def __plot_training_history(self, history):
        ''' Show training history'''
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # График потерь
        axes[1].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def __augment_image(self, image, label):
        return self.data_aug(image, training=True), label

    def recognize_visual(self, tech: bool = False):
        return self.__recognize(tech=tech, show_samples=True)

    def recognize_quick(self, tech: bool = False):
        return self.__recognize(tech=tech, show_samples=False)

    def __get_sample_images_for_class(self, class_name: str, num_samples: int = 9):
        print(f"\n🔍 Search examples for class: '{class_name}'")

        sample_images = []
        searched_paths = []

        # Search in base directory
        base_class_path = os.path.join(self.base_flowers, class_name)
        searched_paths.append(f"📂 Base: {base_class_path}")

        if os.path.exists(base_class_path):
            print(f"   ✅ Found in base directory")
            images = self.__load_sample_images_from_path(base_class_path, num_samples)
            sample_images.extend(images)

        # Search in additional directory
        if hasattr(self, 'add_flowers'):
            add_class_path = os.path.join(self.add_flowers, class_name)
            searched_paths.append(f"📂 Additional: {add_class_path}")

            if os.path.exists(add_class_path):
                print(f"   ✅ Found in additional directory")
                images = self.__load_sample_images_from_path(add_class_path, num_samples)
                sample_images.extend(images)

        # Search in test directory
        if hasattr(self, 'test_data_dir'):
            test_class_path = os.path.join(self.test_data_dir, class_name)
            searched_paths.append(f"📂 Testing: {test_class_path}")

            if os.path.exists(test_class_path):
                print(f"   ✅ Found in test directory")
                images = self.__load_sample_images_from_path(test_class_path, num_samples)
                sample_images.extend(images)

        # Search in all subdirectories recursively
        all_dirs_to_search = [self.base_flowers]
        if hasattr(self, 'add_flowers'):
            all_dirs_to_search.append(self.add_flowers)

        for root_dir in all_dirs_to_search:
            if os.path.exists(root_dir):
                for root, dirs, files in os.walk(root_dir):
                    if class_name in root:
                        print(f"   🔍 Found in subdirectories: {root}")
                        images = self.__load_sample_images_from_path(root, num_samples // 2)
                        sample_images.extend(images)
                        break

        # If not found, search similar names
        if not sample_images:
            print(f"   ⚠️  Class '{class_name}' hasn't found. Search similar")

            # Get all available classes
            available_classes = self.__get_all_available_classes()

            # Search similar names
            similar_classes = []
            for available_class in available_classes:
                if class_name.lower() in available_class.lower() or available_class.lower() in class_name.lower():
                    similar_classes.append(available_class)

            if similar_classes:
                print(f"   🔍 Search similar classes: {similar_classes}")
                # Get first similar class
                similar_class = similar_classes[0]
                similar_path = os.path.join(self.base_flowers, similar_class)
                if not os.path.exists(similar_path) and hasattr(self, 'add_flowers'):
                    similar_path = os.path.join(self.add_flowers, similar_class)

                if os.path.exists(similar_path):
                    print(f"   🔄 Use similar class: '{similar_class}'")
                    images = self.__load_sample_images_from_path(similar_path, num_samples)
                    sample_images.extend(images)

        # Debug details
        if not sample_images:
            print(f"\n❌ Examples haven't found: '{class_name}'")
            print("📋 Was searching in:")
            for path in searched_paths:
                print(f"   {path}")

            # Show available classes
            available_classes = self.__get_all_available_classes()
            if available_classes:
                print(f"\n📋 Avialable classes ({len(available_classes)}):")
                for i, cls in enumerate(sorted(available_classes)[:20]):  # First 20
                    print(f"   {i + 1}. {cls}")
                if len(available_classes) > 20:
                    print(f"   ... and other {len(available_classes) - 20} classes")

            # Check model class list
            if hasattr(self, 'class_names'):
                print(f"\n📋 Classes in model ({len(self.class_names)}):")
                for i, cls in enumerate(self.class_names[:20]):
                    print(f"   {i + 1}. {cls}")
                if len(self.class_names) > 20:
                    print(f"   ... and other {len(self.class_names) - 20} classes")

                # Check class exists
                if class_name in self.class_names:
                    print(f"\n⚠️  Class '{class_name}' exists in model, but not in folders")
                    idx = self.class_names.index(class_name)
                    print(f" Index in model: {idx}")
                else:
                    print(f"\n⚠️  Class '{class_name}' not in model class list")

                    # Search similar in model
                    similar_in_model = [cls for cls in self.class_names
                                        if class_name.lower() in cls.lower() or cls.lower() in class_name.lower()]
                    if similar_in_model:
                        print(f" Similar classes in model: {similar_in_model}")

        return sample_images[:num_samples]

    def __load_sample_images_from_path(self, class_path: str, num_samples: int):
        """Load images from directory"""
        image_files = []

        # Check directory exists
        if not os.path.exists(class_path):
            return []

        # Collect all images
        for file in os.listdir(class_path):
            file_lower = file.lower()
            if file_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                full_path = os.path.join(class_path, file)
                if os.path.isfile(full_path):
                    image_files.append(full_path)

        if not image_files:
            return []

        # Select random images
        try:
            selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        except ValueError:
            selected_files = image_files

        sample_images = []
        for img_path in selected_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((150, 150), Image.Resampling.LANCZOS)
                sample_images.append(img)
            except Exception as e:
                print(f" ⚠️  Error of loading {img_path}: {e}")

        return sample_images

    def __get_all_available_classes(self):
        """Get list of all available classes from all directories"""
        all_classes = set()

        #From base
        if os.path.exists(self.base_flowers):
            for item in os.listdir(self.base_flowers):
                if os.path.isdir(os.path.join(self.base_flowers, item)):
                    all_classes.add(item)

        #From additional
        if hasattr(self, 'add_flowers') and os.path.exists(self.add_flowers):
            for item in os.listdir(self.add_flowers):
                if os.path.isdir(os.path.join(self.add_flowers, item)):
                    all_classes.add(item)

        #From testing
        if hasattr(self, 'test_data_dir') and os.path.exists(self.test_data_dir):
            for item in os.listdir(self.test_data_dir):
                if os.path.isdir(os.path.join(self.test_data_dir, item)):
                    all_classes.add(item)

        return list(all_classes)

    def __load_sample_images(self, class_path: str, num_samples: int = 9):
        """Download images from specified directory"""

        image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

        #Collect all images to directory
        for file in os.listdir(class_path):
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(class_path, file)
                image_files.append(full_path)

        if not image_files:
            print(f"⚠️ No images found in {class_path}")
            return []

        #Get random images
        try:
            selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        except ValueError:
            selected_files = image_files

        sample_images = []
        for img_path in selected_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((150, 150), Image.Resampling.LANCZOS)
                sample_images.append(img)
            except Exception as e:
                print(f"⚠️ Error loading {img_path}: {e}")

        return sample_images

    #Show predictions with examples
    def __show_prediction_with_samples(self, top_class: str, probability: float, samples: list):

        if not samples:
            print(f"\n⚠️ Can't find examples '{top_class}'")
            return

        print("\n" + self.equals_line)
        print(f"🏆 Top class: {top_class}")
        print(f"📊 Confidence: {probability:.2f}%")
        print(f"🖼️ Examples:")
        print(self.equals_line)

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        fig.suptitle(f'Examples: {top_class} (Confidence: {probability:.2f}%)',
                    fontsize=16, fontweight='bold')

        for i, ax in enumerate(axes.flat):
            if i < len(samples):
                ax.imshow(samples[i])
                ax.set_title(f'{top_class} {i + 1}', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    def __create_backup(self):
        """Create backup of model and classes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_folder, f"backup_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)

        #Copy model
        if os.path.exists(self.model_name):
            shutil.copy(self.model_name, os.path.join(backup_path, self.model_name))

        #Copy classes
        if os.path.exists('class_names.json'):
            shutil.copy('class_names.json', os.path.join(backup_path, 'class_names.json'))

        #Save weights
        weights_path = os.path.join(backup_path, "model_weights.weights.h5")
        self.model.save_weights(weights_path)

        print(f"✅ Backup created in {backup_path}")
        return backup_path

    def __create_new_model_with_classes(self, num_classes):
        """Create new model with required count of classes"""
        print(f"Create new model with {num_classes} classes...")

        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3),
            pooling='avg'
        )
        base_model.trainable = False

        new_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        print(f"✅ New model created with {num_classes} classes")
        return new_model

    def evaluate_on_original_data(self, use_test_split=True):
        print("\n" + self.equals_line)
        print("📊 Evaluate on original data")
        print(self.equals_line)

        if not os.path.exists(self.base_flowers):
            print("❌ Data no found")
            return None

        test_dir = self.base_flowers

        try:
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                shuffle=False
            )
        except Exception as e:
            print(f"❌ Dataset creation failure: {e}")
            return None

        # Get classes in alphabet order
        test_class_names = test_ds.class_names
        print(f"\n📋 Classes in test data ({len(test_class_names)}):")
        for i, name in enumerate(test_class_names[:10]):
            print(f"   {i}: {name}")
        if len(test_class_names) > 10:
            print(f" other {len(test_class_names) - 10} classes")

        # Load current classes
        if not hasattr(self, 'class_names') or not self.class_names:
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)

        print(f"📋 Classes count in model: {len(self.class_names)}")
        print(f"📋 First 10 classes: {self.class_names[:10]}")

        # Create mapping from test classes to model classes
        mapping = {}
        for test_class_name in test_class_names:
            if test_class_name in self.class_names:
                mapping[test_class_name] = self.class_names.index(test_class_name)
            else:
                print(f"⚠️ Class '{test_class_name}' from test data doesn't exist in model")

        if not mapping:
            print("❌ There are no common classes between the test data and the model")
            return None

        print(f"\n🗺️  Mapping test classes to model classes:")
        for test_class_name, model_idx in list(mapping.items())[:10]:
            print(f"   '{test_class_name}' → index {model_idx} in model")

        if len(mapping) > 10:
            print(f" other {len(mapping) - 10} classes")

        y_true_all = []
        y_pred_all = []
        total_samples = 0
        correct_samples = 0

        print("\n🔍 Get predictions")
        batch_count = 0

        for images, labels in test_ds:
            predictions = self.model.predict(images, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            for i in range(len(labels)):
                test_class_name = test_class_names[labels[i]]

                if test_class_name in mapping:
                    model_class_idx = mapping[test_class_name]
                    y_true_all.append(model_class_idx)
                    y_pred_all.append(predicted_classes[i])

                    # Check predictions
                    if predicted_classes[i] == model_class_idx:
                        correct_samples += 1
                    total_samples += 1

            batch_count += 1
            if batch_count % 5 == 0:
                print(f"Processed batches: {batch_count}")

        if not y_true_all:
            print("❌ No data to evaluate")
            return None

        # Calculate accuracy
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        print(f"\n✅ Accuracy on base data: {accuracy:.4f}")
        print(f"📊 Samples evaluated: {total_samples}")
        print(f"📈 Correctly classified: {correct_samples}/{total_samples}")

        print("\n📋 Classes analys (old classes):")

        class_stats = {}
        for test_class_name, model_idx in mapping.items():
            mask = (y_true_all == model_idx)
            if np.sum(mask) > 0:
                class_correct = np.sum(y_pred_all[mask] == model_idx)
                class_total = np.sum(mask)
                class_accuracy = class_correct / class_total if class_total > 0 else 0.0

                if test_class_name in self.base_flowers:
                    class_type = "old"
                else:
                    class_type = "new"

                class_stats[test_class_name] = {
                    'accuracy': class_accuracy,
                    'total': class_total,
                    'correct': class_correct,
                    'type': class_type
                }

        # Sort by accuracy
        sorted_stats = sorted(class_stats.items(), key=lambda x: x[1]['accuracy'])

        print("\n📊 Best classes:")
        for class_name, stats in sorted_stats[-5:]:
            print(f"   🏆 {class_name} ({stats['type']}): {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")

        print("\n📉 Worst classes:")
        for class_name, stats in sorted_stats[:5]:
            print(
                f"   ⚠️  {class_name} ({stats['type']}): {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")

        try:
            if len(mapping) > 0:
                print("\n🎯 Error matrix for old classes:")

                class_indices = list(mapping.values())
                class_names = list(mapping.keys())

                cm = confusion_matrix(y_true_all, y_pred_all, labels=class_indices)

                plt.figure(figsize=(max(12, len(class_names)), max(10, len(class_names))))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names,
                            yticklabels=class_names)
                plt.title('Error matrix on base data')
                plt.ylabel('True classes')
                plt.xlabel('Predicted classes')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"⚠️ Error while creating error matrix: {e}")

        return accuracy

    def evaluate_on_new_data(self, use_test_split=True):
        print("\n" + self.equals_line)
        print("📊 Evaluate on new data")
        print(self.equals_line)

        if not os.path.exists(self.add_flowers):
            print("❌ New data not found")
            return None

        # Download current classes
        if not hasattr(self, 'class_names') or not self.class_names:
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)

        print(f"📋 All classes in model: {len(self.class_names)}")

        # Get list of all new classes
        test_class_names = [d for d in os.listdir(self.add_flowers)
                            if os.path.isdir(os.path.join(self.add_flowers, d))]

        if not test_class_names:
            print("❌ No classes in new data")
            return None

        print(f"📋 Classes for evaluate ({len(test_class_names)}):")
        for i, cls in enumerate(test_class_names):
            if cls in self.class_names:
                idx = self.class_names.index(cls)
                print(f"   {i}. {cls} → index {idx} in model")
            else:
                print(f"   {i}. {cls} → ❌ doesn't exist in model")

        # Creating a dataset from new data
        try:
            test_ds = tf.keras.utils.image_dataset_from_directory(
                self.add_flowers,
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                shuffle=False
            )
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            return None

        dataset_class_names = test_ds.class_names
        print(f"\n📋 Classes in the dataset (alphabetical order):")
        for i, cls in enumerate(dataset_class_names):
            print(f"   {i}: {cls}")

        mapping = {}
        for dataset_class_name in dataset_class_names:
            if dataset_class_name in self.class_names:
                mapping[dataset_class_name] = self.class_names.index(dataset_class_name)
            else:
                print(f"⚠️ Class '{dataset_class_name}' not in model")

        if not mapping:
            print("❌ There are no common classes between the new data and the model")
            return None

        print("\n🔍 Get predictions")
        y_true_all = []
        y_pred_all = []
        total_samples = 0
        correct_samples = 0

        for batch_idx, (images, labels) in enumerate(test_ds):
            predictions = self.model.predict(images, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)

            for i in range(len(labels)):
                dataset_class_name = dataset_class_names[labels[i]]

                if dataset_class_name in mapping:
                    model_class_idx = mapping[dataset_class_name]
                    y_true_all.append(model_class_idx)
                    y_pred_all.append(predicted_classes[i])

                    if predicted_classes[i] == model_class_idx:
                        correct_samples += 1
                    total_samples += 1

            if (batch_idx + 1) % 3 == 0:
                print(f"Processed batches: {batch_idx + 1}")

        if not y_true_all:
            print("❌ No data for evaluate")
            return None

        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        print(f"\n✅ Accuracy on new data: {accuracy:.4f}")
        print(f"📊 Samples evaluated: {total_samples}")
        print(f"📈 Correctly classified: {correct_samples}/{total_samples}")

        print("\n📋 Detailed report on new classes:")

        for dataset_class_name, model_idx in mapping.items():
            mask = np.array(y_true_all) == model_idx
            if np.sum(mask) > 0:
                class_correct = np.sum(np.array(y_pred_all)[mask] == model_idx)
                class_total = np.sum(mask)
                class_accuracy = class_correct / class_total if class_total > 0 else 0.0

                if dataset_class_name in test_class_names:
                    if dataset_class_name in self.class_names:
                        try:
                            with open('original_class_names.json', 'r') as f:
                                original_classes = json.load(f)
                            if dataset_class_name in original_classes:
                                class_type = "old"
                            else:
                                class_type = "new"
                        except:
                            class_type = "unknown"
                    else:
                        class_type = "not in model"
                else:
                    class_type = "test"

                status = "✅" if class_accuracy > 0.7 else "⚠️" if class_accuracy > 0.3 else "❌"
                print(
                    f"   {status} {dataset_class_name} ({class_type}): {class_accuracy:.3f} ({class_correct}/{class_total})")

        try:
            if len(mapping) > 0:
                print("\n🎯 Error matrix for new data:")

                class_indices = list(mapping.values())
                class_names = list(mapping.keys())

                cm = confusion_matrix(y_true_all, y_pred_all, labels=class_indices)

                plt.figure(figsize=(max(10, len(class_names)), max(8, len(class_names))))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                            xticklabels=class_names,
                            yticklabels=class_names)
                plt.title('Error matrix for new data')
                plt.ylabel('True classes')
                plt.xlabel('Predicted classes')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"⚠️ Error while creating error matrix: {e}")

        return accuracy

    def evaluate_on_both_datasets(self):
        print("\n" + self.equals_line)
        print("📊 Full evaluate")
        print(self.equals_line)

        results = {}
        details = {}

        if not hasattr(self, 'class_names') or not self.class_names:
            with open('class_names.json', 'r') as f:
                self.class_names = json.load(f)


        original_classes = []
        if os.path.exists('original_class_names.json'):
            with open('original_class_names.json', 'r') as f:
                original_classes = json.load(f)

        print(f"📋 All classes in model: {len(self.class_names)}")
        print(f"📋 Source classes: {len(original_classes)}")
        print(f"📋 New classes: {len(self.class_names) - len(original_classes)}")

        old_classes = [cls for cls in self.class_names if cls in original_classes]
        new_classes = [cls for cls in self.class_names if cls not in original_classes]

        print(f"\n📊 Class distribution:")
        print(f"Old classes: {len(old_classes)}")
        print(f"New classes: {len(new_classes)}")

        if old_classes:
            print(f"   📌 Examples of old classes: {old_classes[:5]}")
        if new_classes:
            print(f"   🆕 Examples of new classes: {new_classes[:5]}")

        if os.path.exists(self.base_flowers):
            print("\n" + self.equals_line)
            print("📊 Evaluate on source data")
            print(self.equals_line)

            original_acc = self.evaluate_on_original_data()
            results['original_data'] = original_acc

            details['old_classes_in_original'] = len(old_classes)
            details['new_classes_in_original'] = len(new_classes)

        if os.path.exists(self.add_flowers) and os.listdir(self.add_flowers):
            print("\n" + self.equals_line)
            print("📊 Evaluate on new data")
            print(self.equals_line)

            new_acc = self.evaluate_on_new_data()
            results['new_data'] = new_acc

            if os.path.exists(self.add_flowers):
                new_data_classes = [d for d in os.listdir(self.add_flowers)
                                    if os.path.isdir(os.path.join(self.add_flowers, d))]
                old_in_new = [cls for cls in new_data_classes if cls in old_classes]
                new_in_new = [cls for cls in new_data_classes if cls in new_classes]
                details['old_classes_in_new_data'] = len(old_in_new)
                details['new_classes_in_new_data'] = len(new_in_new)

        print("\n" + self.equals_line)
        print("📈 Summary")
        print(self.equals_line)

        for dataset_name, accuracy in results.items():
            if accuracy is not None:
                print(f"{dataset_name}: {accuracy:.4f}")

        # Overall score (weighted by number of classes)
        if 'original_data' in results and 'new_data' in results:
            total_classes = len(self.class_names)
            old_weight = len(old_classes) / total_classes if total_classes > 0 else 0
            new_weight = len(new_classes) / total_classes if total_classes > 0 else 0

            weighted_acc = (results['original_data'] * old_weight +
                            results['new_data'] * new_weight)

            print(f"\n🎯 Weighted precision: {weighted_acc:.4f}")
            print(f"Contribution of old classes ({len(old_classes)}): {results['original_data']:.4f} * {old_weight:.2f}")
            print(f"Contribution of new classes ({len(new_classes)}): {results['new_data']:.4f} * {new_weight:.2f}")

        print(f"\n📋 Classes detail:")
        print(f"All classes in model: {len(self.class_names)}")
        print(f"Old classes: {len(old_classes)}")
        print(f"New classes: {len(new_classes)}")

        if 'original_data' in results and original_classes:
            print(f"\n⚠️  Testing catastrophic forgetting:")
            print(f" Accuracy on old classes: {results['original_data']:.4f}")

            if results['original_data'] < 0.5:
                print(f"   ❌ High risk of catastrophic forgetting!")
            elif results['original_data'] < 0.7:
                print(f"   ⚠️ Moderate risk of forgetting")
            else:
                print(f"   ✅ New classes are well preserved")

        return results

    def __split_data_into_folders(self, source_dir, train_dir, test_dir, test_size=0.1, val_size=0.2, seed=42):

        print(f"📂 Split data from {source_dir}...")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        temp_train_dir = os.path.join(train_dir, 'temp_train')
        temp_val_dir = os.path.join(train_dir, 'temp_val')
        temp_test_dir = test_dir

        os.makedirs(temp_train_dir, exist_ok=True)
        os.makedirs(temp_val_dir, exist_ok=True)
        os.makedirs(temp_test_dir, exist_ok=True)

        total_images = 0

        for class_name in os.listdir(source_dir):
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Get all images from class
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(Path(class_dir).glob(ext))
                images.extend(Path(class_dir).glob(ext.upper()))

            if len(images) < 5:# Skipping a class with a small number of images
                print(f"⚠️ Class {class_name} has less than 5 images, skip it")
                continue

            image_paths = [str(img) for img in images]
            total_images += len(image_paths)

            train_val_paths, test_paths = train_test_split(
                image_paths, test_size=test_size, random_state=seed, shuffle=True
            )

            train_paths, val_paths = train_test_split(
                train_val_paths, test_size=val_size / (1 - test_size), random_state=seed, shuffle=True
            )

            os.makedirs(os.path.join(temp_train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(temp_val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(temp_test_dir, class_name), exist_ok=True)

            for img_path in train_paths:
                img_name = os.path.basename(img_path)
                dst_path = os.path.join(temp_train_dir, class_name, img_name)
                shutil.copy(img_path, dst_path)

            for img_path in val_paths:
                img_name = os.path.basename(img_path)
                dst_path = os.path.join(temp_val_dir, class_name, img_name)
                shutil.copy(img_path, dst_path)

            for img_path in test_paths:
                img_name = os.path.basename(img_path)
                dst_path = os.path.join(temp_test_dir, class_name, img_name)
                shutil.copy(img_path, dst_path)

            print(f"   {class_name}: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

        print(f"📊 Total images processed: {total_images}")
        print(f"📁 The structure was created in {train_dir} and {test_dir}")

        return temp_train_dir, temp_val_dir, temp_test_dir

    def debug_model_output(self):

        print("\n🔍 Debug model")

        print(f"📐 Model architecture:")
        print(f" Input size: {self.img_height}x{self.img_width}x3")

        if hasattr(self.model.layers[-1], 'output_shape'):
            print(f" Output classes: {self.model.layers[-1].output_shape[-1]}")

        output_layer = self.model.layers[-1]
        if hasattr(output_layer, 'units'):
            print(f" Neurons in the output layer: {output_layer.units}")

        if hasattr(output_layer, 'activation'):
            if callable(output_layer.activation):
                print(f"Output layer activation: {output_layer.activation.__name__}")
            else:
                print(f"Output layer activation: {output_layer.activation}")

        if hasattr(self, 'class_names'):
            print(f"\n📋 Class matching:")
            print(f" Classes in model: {output_layer.units if hasattr(output_layer, 'units') else 'unknown'}")
            print(f" Classes in class_names: {len(self.class_names)}")

            if hasattr(output_layer, 'units'):
                if output_layer.units == len(self.class_names):
                    print(f"   ✅ The number of classes corresponds")
                else:
                    print(f"   ❌ inconsistency {output_layer.units} vs class_names {len(self.class_names)}")

            print(f"\n📋 First 10 classes:")
            for i, cls in enumerate(self.class_names[:10]):
                print(f"   {i}: {cls}")

            available_classes = self.__get_all_available_classes()
            print(f"\n📂 Available classes in data: {len(available_classes)}")

            missing_in_data = [cls for cls in self.class_names if cls not in available_classes]
            if missing_in_data:
                print(f"   ⚠️  Classes in model, but not in data ({len(missing_in_data)}):")
                for cls in missing_in_data[:10]:
                    print(f"      - {cls}")
                if len(missing_in_data) > 10:
                    print(f" and other {len(missing_in_data) - 10}")

        print(f"\n🧪 Test run:")
        test_input = np.random.randn(1, self.img_height, self.img_width, 3).astype(np.float32)
        test_output = self.model.predict(test_input, verbose=0)

        print(f" Output shape: {test_output.shape}")
        print(f" Value range: [{test_output.min():.6f}, {test_output.max():.6f}]")
        print(f" Sum of probabilities: {test_output.sum():.6f}")

    def incremental_learning_with_diagnostics(self):
        """
        Incremental learning on new data with an increased amount of old data
        """
        print("\n" + self.equals_line)
        print("🔄 Incremental learning")
        print(self.equals_line)

        self.__create_backup()

        # Loading current classes
        if not os.path.exists('class_names.json'):
            print("❌ File class_names.json not found!")
            return None

        with open('class_names.json', 'r') as f:
            current_classes = json.load(f)

        print(f"📋 Current classes: {len(current_classes)}")

        # Analys of new data
        if not os.path.exists(self.add_flowers):
            print(f"❌ Folder {self.add_flowers} doesn't exist")
            return None

        new_classes = []
        for d in os.listdir(self.add_flowers):
            if os.path.isdir(os.path.join(self.add_flowers, d)):
                new_classes.append(d)

        if not new_classes:
            print("❌ No new classes for learning")
            return None

        truly_new_classes = [cls for cls in new_classes if cls not in current_classes]
        existing_classes = [cls for cls in new_classes if cls in current_classes]

        print(f"\n📊 Statistics:")
        print(f" Total new classes: {len(new_classes)}")
        print(f" Already existing: {len(existing_classes)}")
        print(f" Absolutely new: {len(truly_new_classes)}")

        if truly_new_classes:
            print(f"   🆕 New classes: {truly_new_classes}")

        # Create a combined list of classes
        all_classes = current_classes.copy()
        for new_class in truly_new_classes:
            if new_class not in all_classes:
                all_classes.append(new_class)

        print(f"\n📈 There will be a total of classes: {len(all_classes)}")

        # Creating temporary directories with part of old data
        temp_dir = 'Plants/incremental_temp'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            time.sleep(1)

        train_dir = os.path.join(temp_dir, 'train')
        val_dir = os.path.join(temp_dir, 'val')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        print(f"\n📁 Create dataset:")

        print("   📥 loading old data:")
        old_samples_total = 0

        for class_name in current_classes:
            old_class_dir = os.path.join(self.base_flowers, class_name)

            if not os.path.exists(old_class_dir):
                continue

            all_images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                for img in os.listdir(old_class_dir):
                    if img.lower().endswith(ext):
                        all_images.append(os.path.join(old_class_dir, img))

            if not all_images:
                continue

            # Get 200 images
            n_samples = min(200, len(all_images))
            if len(all_images) < 100:
                n_samples = len(all_images)

            selected_images = random.sample(all_images, n_samples) if n_samples < len(all_images) else all_images

            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            split_idx = int(len(selected_images) * 0.8)
            train_images = selected_images[:split_idx]
            val_images = selected_images[split_idx:]

            for img_path in train_images:
                if os.path.exists(img_path):
                    dst = os.path.join(train_class_dir, os.path.basename(img_path))
                    try:
                        shutil.copy(img_path, dst)
                    except Exception as e:
                        print(f"⚠️ Copy error {img_path}: {e}")

            for img_path in val_images:
                if os.path.exists(img_path):
                    dst = os.path.join(val_class_dir, os.path.basename(img_path))
                    try:
                        shutil.copy(img_path, dst)
                    except Exception as e:
                        print(f"⚠️ Copy error {img_path}: {e}")

            old_samples_total += n_samples
            print(f"📦 {class_name}: {n_samples} images")

        print(f"📊 Total old images: {old_samples_total}")

        print("\n   📥 Loading new data:")
        new_samples_total = 0

        for class_name in new_classes:
            new_class_dir = os.path.join(self.add_flowers, class_name)

            if not os.path.exists(new_class_dir):
                continue

            all_images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                for img in os.listdir(new_class_dir):
                    if img.lower().endswith(ext):
                        all_images.append(os.path.join(new_class_dir, img))

            if not all_images:
                continue

            n_samples = len(all_images)

            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            if n_samples >= 5:
                split_idx = int(len(all_images) * 0.8)
                train_images = all_images[:split_idx]
                val_images = all_images[split_idx:]
            else:
                train_images = all_images
                val_images = []

            for img_path in train_images:
                if os.path.exists(img_path):
                    dst = os.path.join(train_class_dir, os.path.basename(img_path))
                    try:
                        shutil.copy(img_path, dst)
                    except Exception as e:
                        print(f"⚠️ Copy error {img_path}: {e}")

            for img_path in val_images:
                if os.path.exists(img_path):
                    dst = os.path.join(val_class_dir, os.path.basename(img_path))
                    try:
                        shutil.copy(img_path, dst)
                    except Exception as e:
                        print(f"⚠️ Copy error {img_path}: {e}")

            new_samples_total += n_samples
            print(f"      🆕 {class_name}: {n_samples} images")

        print(f"   📊 Total new images: {new_samples_total}")
        print(f"   📈 Total images: {old_samples_total + new_samples_total}")

        if old_samples_total == 0 and new_samples_total == 0:
            print("❌ No data for learning")
            return None

        print("\n🎯 Create dataset")

        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=32,
                shuffle=True
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                seed=123,
                image_size=(self.img_height, self.img_width),
                batch_size=32,
                shuffle=False
            )

            # Get classes from the dataset (important for correct labeling)
            dataset_class_names = train_ds.class_names
            print(f"\n📋 Classes in dataset ({len(dataset_class_names)}):")
            for i, cls in enumerate(dataset_class_names):
                print(f"   {i}: {cls}")

            self.class_names = dataset_class_names

            with open('class_names.json', 'w') as f:
                json.dump(self.class_names, f)

            print(f"💾 Class list has been saved.")

        except Exception as e:
            print(f"❌ Error creating datasets: {e}")
            return None

        print("\n⚖️  Calculating class weights for balancing")
        class_weights = {}

        # Get the number of images by class
        class_counts = {}
        for class_name in self.class_names:
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                class_counts[class_name] = count
                print(f"   {class_name}: {count} изображений")

        # Calculate weights
        max_count = max(class_counts.values()) if class_counts else 1
        for i, class_name in enumerate(self.class_names):
            if class_name in class_counts:
                if class_name in current_classes:
                    # Old classes get more weight
                    weight = max_count / class_counts[class_name] * 1.5  # Coefficient 1.5 for old
                else:
                    # New classes get normal weight
                    weight = max_count / class_counts[class_name] * 0.7  # Coefficient 0.7 for new
                class_weights[i] = min(weight, 3.0)  # Limit the maximum weight
            else:
                class_weights[i] = 1.0

        print(f"\n📊 Classes weights:")
        print(f"   • Old classes: increased weight (×1.5)")
        print(f"   • New classes: reduced weight (×0.7)")

        # Extending the model (if needed)
        old_num_classes = len(current_classes)
        new_num_classes = len(self.class_names)

        if new_num_classes > old_num_classes:
            print(f"\n🔧 Extending the output layer: {old_num_classes} -> {new_num_classes}")

            old_weights = self.model.get_weights()

            new_model = self.__create_new_model_with_classes(new_num_classes)

            if len(old_weights) >= 2:
                new_weights = new_model.get_weights()

                # Copy all weights except the last layer
                copy_layers = min(len(old_weights), len(new_weights)) - 2

                for i in range(copy_layers):
                    if i < len(old_weights) and i < len(new_weights):
                        if old_weights[i].shape == new_weights[i].shape:
                            new_weights[i] = old_weights[i]

                # Copy the output layer for the old classes
                old_w, old_b = old_weights[-2], old_weights[-1]
                new_w, new_b = new_weights[-2], new_weights[-1]

                # Creating a mapping of old indexes
                for i, old_class in enumerate(current_classes):
                    if old_class in self.class_names:
                        new_idx = self.class_names.index(old_class)
                        new_w[:, new_idx] = old_w[:, i]
                        new_b[new_idx] = old_b[i]

                new_model.set_weights(new_weights)

            self.model = new_model
            print("✅ The model has been extended")

        print("\n⚙️  Setting up the training")

        # Freezing base layers
        for layer in self.model.layers:
            if 'mobilenet' in layer.name.lower():
                layer.trainable = False

        # Defrost the last layers
        for layer in self.model.layers[-3:]:
            layer.trainable = True

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("   ❄️ Base layers are frozen")
        print("   🔥 The last 3 layers are defrosted")
        print("   📉 Low learning rate: 0.00001")

        print("\n🚀 Start training")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]

        try:
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=60,
                callbacks=callbacks,
                verbose=1
            )

            print("\n✅ Training completed")

        except Exception as e:
            print(f"❌ Training error: {e}")
            return None

        self.model.save(self.model_name)
        print(f"💾 Model saved: {self.model_name}")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

        print("\n" + self.equals_line)
        print("🎯 Incremental training completed")
        print(self.equals_line)

        return history





