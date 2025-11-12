from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Train BERT model and save to data/models/bert'

    def add_arguments(self, parser):
        parser.add_argument('--model', default='distilbert-base-uncased',
                          help='Model name (default: distilbert-base-uncased, faster than bert-base)')
        parser.add_argument('--epochs', type=int, default=1, 
                          help='Number of training epochs (default: 1)')
        parser.add_argument('--batch_size', type=int, default=32, 
                          help='Batch size (default: 32, increase for faster training)')
        parser.add_argument('--max_len', type=int, default=128, 
                          help='Maximum sequence length (default: 128, shorter = faster)')
        parser.add_argument('--max_samples', type=int, default=500, 
                          help='Maximum number of samples to use (default: 500, set to 0 for all)')

    def handle(self, *args, **options):
        # Lazy import to avoid importing TensorFlow/Transformers at module import time
        from detector.dl.train_bert import run as run_bert
        # Default to 500 samples if not specified or if 0 is passed
        max_samples = options.get('max_samples', 500)
        if max_samples <= 0:
            max_samples = 500
        run_bert(
            model_name=options['model'],
            epochs=options['epochs'],
            batch_size=options['batch_size'],
            max_len=options['max_len'],
            max_samples=max_samples,
        )
        self.stdout.write(self.style.SUCCESS('BERT training completed.'))


