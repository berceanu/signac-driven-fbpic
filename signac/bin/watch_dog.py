"""
Watch folder for changes. 
Usage: python bin/watch_dog.py <dir>
"""
import sys
import time
import logging
import pathlib
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class MyHandler(PatternMatchingEventHandler):
    def process(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        p = pathlib.Path(event.src_path)
        fn = p.name
        if (not fn.startswith("eci")) and (not fn.startswith("fort")):
            logging.info("%s : %s" % (p, event.event_type))

    def on_created(self, event):
        self.process(event)

    def on_deleted(self, event):
        self.process(event)

    def on_modified(self, event):
        self.process(event)

    def on_moved(self, event):
        self.process(event)


if __name__ == "__main__":
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    path = sys.argv[1] if len(sys.argv) > 1 else "."

    event_handler = MyHandler(ignore_directories=True)
    observer = Observer()

    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
