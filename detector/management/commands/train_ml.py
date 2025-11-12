from django.core.management.base import BaseCommand
from detector.ml.train_ml import run as run_ml


class Command(BaseCommand):
    help = 'Train ML models (NB, RF, SVM) and save to data/models'

    def add_arguments(self, parser):
        parser.add_argument('--max_samples', type=int, default=1000, 
                          help='Maximum number of samples to use (default: 1000, set to 0 for all)')

    def handle(self, *args, **options):
        # Default to 1000 samples if not specified or if 0 is passed
        max_samples = options.get('max_samples', 1000)
        if max_samples <= 0:
            max_samples = None  # Use all samples
        run_ml(max_samples=max_samples)
        self.stdout.write(self.style.SUCCESS('ML training completed.'))


