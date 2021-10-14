import glob
import os

from google.cloud import storage

MODEL_BUCKET = "shopping-classifier-test"
MODEL_DIRECTORY = "model"


class CloudStorageDao:
    def __init__(self):
        self.client = storage.Client(project='shopping-classifier')

    @staticmethod
    def _copy_local_directory_to_gcs(local_path, bucket, gcs_path):
        """Recursively copy a directory of files to GCS.

        local_path should be a directory and not have a trailing slash.
        """
        assert os.path.isdir(local_path)
        for local_file in glob.glob(local_path + '/**'):
            if not os.path.isfile(local_file):
                continue
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

    def copy_to_bucket(self, tmp_dir_name):
        bucket = self.client.get_bucket(MODEL_BUCKET)
        self._copy_local_directory_to_gcs(tmp_dir_name, bucket, MODEL_DIRECTORY)
