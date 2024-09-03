import argparse
from SRGAN.Data import DatasetModifier
from SRGAN.Generator import Generator
from SRGAN.Discriminator import Discriminator
from SRGAN.LossUtils import VGGFeatureExtractor
from SRGAN.Trainers import PreTrainer, AdversarialTrainer


def load_dataset():
    print("Loading and modifying dataset...")
    dataset_modifier = DatasetModifier()
    train_loader, valid_loader = dataset_modifier.get_loaders()
    print("Dataset loaded and modified successfully.")
    return train_loader, valid_loader


def pre_train(generator, train_loader):
    print("Starting pre-training...")
    trainer = PreTrainer(generator)
    trainer.train(train_loader)
    print("Pre-training completed.")
    return trainer


def adversarial_train(generator, discriminator, train_loader):
    print("Starting adversarial training...")
    advtrainer = AdversarialTrainer(
        generator, discriminator)
    advtrainer.train(train_loader)
    print("Adversarial training completed.")
    return advtrainer


def export_model(trainer, path, key='pretrain'):
    print(f"Exporting model to {path}...")
    if key == 'pretrain':
        trainer.save_model_script(path)
    elif key == 'advtrain':
        trainer.save_models_script(path)
    else:
        raise ValueError("Invalid key. Choose from 'pretrain' or 'advtrain'")
    print(f"Model exported successfully to {path}.")


def main():
    parser = argparse.ArgumentParser(description="SRGAN Training Script")
    parser.add_argument('--export', type=str,
                        help='Path to export the trained model', required=True)
    parser.add_argument('--mode', type=str, choices=['pretrain', 'advtrain', 'both'],
                        default='both', help='Choose the training mode: pretrain, advtrain, or both')

    args = parser.parse_args()

    generator = Generator()
    discriminator = Discriminator()
    vgg_feature_extractor = VGGFeatureExtractor()

    train_loader, _ = load_dataset()

    if args.mode == 'pretrain':
        pre_trainer = pre_train(generator, train_loader)
        export_model(pre_trainer, args.export)
    elif args.mode == 'advtrain':
        adv_trainer = adversarial_train(generator, discriminator,
                                        train_loader, vgg_feature_extractor)
        export_model(adv_trainer, args.export, key='advtrain')
    elif args.mode == 'both':
        pre_trainer = pre_train(generator, train_loader)
        export_model(pre_trainer, args.export)
        adv_trainer = adversarial_train(generator, discriminator,
                                        train_loader, vgg_feature_extractor)
        export_model(adv_trainer, args.export, key='advtrain')
    else:
        raise ValueError(
            "Invalid mode. Choose from 'pretrain', 'advtrain', or 'both'")


if __name__ == '__main__':
    main()
