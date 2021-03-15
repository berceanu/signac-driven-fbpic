import dropbox
import pathlib
import os
import contextlib
import time
import datetime
import argparse

@contextlib.contextmanager
def stopwatch(message):
    """Context manager to print how long a block of code took."""
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print('Total elapsed time for %s: %.3f' % (message, t1 - t0))


def upload(path_to_file, dbx=dropbox.Dropbox('F79X1kIK8MEAAAAAAAAAAWkcCbtuJk62bZYqANCW1ZWJmC2E1LPXzuZ8ZryCJrwS'), overwrite=True):
    """Upload a file.
    Return the request response, or None in case of error.
    """
    db_path = '/' + path_to_file.name
    mode = (dropbox.files.WriteMode.overwrite
            if overwrite
            else dropbox.files.WriteMode.add)
    mtime = os.path.getmtime(path_to_file)
    with open(path_to_file, 'rb') as f:
        data = f.read()
    with stopwatch('upload %d bytes' % len(data)):
        try:
            res = dbx.files_upload(
                data, db_path, mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True)
        except dropbox.exceptions.ApiError as err:
            print('*** API error', err)
            return None
    print('uploaded as', res.name.encode('utf8'))
    return res



def main():
    """Main entry point."""
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Upload a file to Dropbox')

    # Add the arguments
    my_parser.add_argument('Path',
                        metavar='path',
                        type=str,
                        help='the path to file')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = pathlib.Path(args.Path)

    upload(input_path)


if __name__ == '__main__':
    main()
