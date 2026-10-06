from pathlib import Path


class RunLock:
    """Hold a nonblocking writer lock that the OS releases on process exit."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Another process owns: " + str(self.path)) from None
        return self

    def __exit__(self, *args):
        import fcntl

        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()
