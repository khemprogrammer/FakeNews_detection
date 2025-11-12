from django.core.management.base import BaseCommand
from detector.dl.train_lstm import run as run_lstm


class Command(BaseCommand):
    help = 'Train LSTM model and save to data/models'

    def handle(self, *args, **options):
        run_lstm()
        self.stdout.write(self.style.SUCCESS('LSTM training completed.'))


