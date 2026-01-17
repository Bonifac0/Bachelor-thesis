import time


class ETA:
    def __init__(self, total: int) -> None:
        self.start_time = time.time()
        self.total = total

    def print_eta(self, current: int) -> str:
        elapsed = time.time() - self.start_time
        avg_time = elapsed / current
        remaining_batches = self.total - current
        eta_seconds = remaining_batches * avg_time

        # Format ETA into hh:mm:ss
        hrs, rem = divmod(int(eta_seconds), 3600)
        mins, secs = divmod(rem, 60)
        eta_formatted = f"{hrs:02}:{mins:02}:{secs:02}"
        return f"| ETA: {eta_formatted}"
