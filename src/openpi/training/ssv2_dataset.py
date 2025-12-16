import json
import os
import datasets
import numpy as np

_CITATION = """
@inproceedings{goyal2017something,
  title={The" something something" video database for learning and evaluating visual common sense},
  author={Goyal, Raghav and Ebrahimi Kahou, Samira and Michalski, Vincent and Materzynska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yianilos, Peter and Mueller-Freitag, Moritz and others},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5842--5850},
  year={2017}
}
"""

_DESCRIPTION = """\
The Something-Something dataset (version 2) is a collection of 220,847 labeled video clips of humans performing pre-defined, basic actions with everyday objects. It is designed to train machine learning models in fine-grained understanding of human hand gestures like putting something into something, turning something upside down and covering something with something.
"""


class SomethingSomethingV2(datasets.GeneratorBasedBuilder):
    """Charades is dataset composed of 9848 videos of daily indoors activities collected through Amazon Mechanical Turk"""

    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "video_id": datasets.Value("string"),
                    "video": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "label": datasets.Value(
                        "string"
                    ),  # Changed from ClassLabel to string
                    "placeholders": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    @property
    def manual_download_instructions(self):
        return (
            "To use Something-Something-v2, please download the 19 data files and the labels file "
            "from 'https://developer.qualcomm.com/software/ai-datasets/something-something'. "
            "Unzip the 19 files and concatenate the extracts in order into a tar file named '20bn-something-something-v2.tar.gz. "
            "Use command like `cat 20bn-something-something-v2-?? >> 20bn-something-something-v2.tar.gz` "
            "Place the `labels.zip` file and the tar file into a folder '/path/to/data/' and load the dataset using "
            "`load_dataset('something-something-v2', data_dir='/path/to/data')`"
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.manual_dir
        labels_path = os.path.join(data_dir, "labels.zip")
        videos_path = os.path.join(data_dir, "20bn-something-something-v2.tar.gz")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"labels.zip doesn't exist in {data_dir}. Please follow manual download instructions."
            )

        if not os.path.exists(videos_path):
            raise FileNotFoundError(
                f"20bn-something-sokmething-v2.tar.gz doesn't exist in {data_dir}. Please follow manual download instructions."
            )

        labels_path = dl_manager.extract(labels_path)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": os.path.join(
                        labels_path, "labels", "train.json"
                    ),
                    "video_files": dl_manager.iter_archive(videos_path),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": os.path.join(
                        labels_path, "labels", "validation.json"
                    ),
                    "video_files": dl_manager.iter_archive(videos_path),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file": os.path.join(labels_path, "labels", "test.json"),
                    "video_files": dl_manager.iter_archive(videos_path),
                    "labels_file": os.path.join(
                        labels_path, "labels", "test-answers.csv"
                    ),
                },
            ),
        ]

    def _generate_examples(self, annotation_file, video_files, labels_file=None):
        data = {}
        labels = None
        if labels_file is not None:
            with open(labels_file, "r", encoding="utf-8") as fobj:
                labels = {}
                for label in fobj.readlines():
                    label = label.strip().split(";")
                    labels[label[0]] = label[1]

        with open(annotation_file, "r", encoding="utf-8") as fobj:
            annotations = json.load(fobj)
            for annotation in annotations:
                if "template" in annotation:
                    annotation["template"] = (
                        annotation["template"].replace("[", "").replace("]", "")
                    )
                if labels:
                    annotation["template"] = labels[annotation["id"]]
                data[annotation["id"]] = annotation

        idx = 0
        for path, file in video_files:
            video_id = os.path.splitext(os.path.split(path)[1])[0]

            if video_id not in data:
                continue

            info = data[video_id]
            yield idx, {
                "video_id": video_id,
                "video": file,
                "placeholders": info.get("placeholders", []),
                "label": info["label"] if "label" in info else -1,
                "text": info["template"],
            }

            idx += 1
