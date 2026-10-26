import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class SDSDataset(BaseDataset):
    """ SeaDronesSea dataset
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.seadronesee_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        ext = sequence_info['ext']
        startFrame = sequence_info['startFrame']
        endFrame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame}.{ext}'.format(base_path=self.base_path,sequence_path=sequence_path, frame=frame_num,
                                                                        ext=ext) for frame_num in range(startFrame + init_omit, endFrame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'seadronessee', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            { "name": "1", "path": "Compressed/test",
                "startFrame": 486,"endFrame": 636,"ext": "jpg","anno_path": "test_annotations_first_frame/1.txt","object_class": "vehicle"},
            { "name": "2", "path": "Compressed/test",
                "startFrame": 494,"endFrame": 646,"ext": "jpg","anno_path": "test_annotations_first_frame/2.txt","object_class": "vehicle"},
            { "name": "3", "path": "Compressed/test",
                "startFrame": 503,"endFrame": 668,"ext": "jpg","anno_path": "test_annotations_first_frame/3.txt","object_class": "vehicle"},
            { "name": "4","path": "Compressed/test",
                "startFrame": 518,"endFrame": 688,"ext": "jpg","anno_path": "test_annotations_first_frame/4.txt","object_class": "vehicle"},
            { "name": "5","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1696,"ext": "jpg","anno_path": "test_annotations_first_frame/5.txt","object_class": "vehicle"},
            { "name": "6","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1676,"ext": "jpg","anno_path": "test_annotations_first_frame/6.txt","object_class": "vehicle" },
            { "name": "7","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1696,"ext": "jpg","anno_path": "test_annotations_first_frame/7.txt","object_class": "vehicle"},
            { "name": "8","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1696,"ext": "jpg","anno_path": "test_annotations_first_frame/8.txt","object_class": "vehicle"},
            { "name": "9","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1696,"ext": "jpg","anno_path": "test_annotations_first_frame/9.txt","object_class": "vehicle"},
            { "name": "10","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1696,"ext": "jpg","anno_path": "test_annotations_first_frame/10.txt","object_class": "vehicle"},
            { "name": "11","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1660,"ext": "jpg","anno_path": "test_annotations_first_frame/11.txt","object_class": "vehicle"},
            { "name": "12","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1696,"ext": "jpg","anno_path": "test_annotations_first_frame/12.txt","object_class": "vehicle"},
            { "name": "13","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1625,"ext": "jpg","anno_path": "test_annotations_first_frame/13.txt","object_class": "vehicle"},
            { "name": "14","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1636,"ext": "jpg","anno_path": "test_annotations_first_frame/14.txt","object_class": "vehicle"},
            { "name": "15","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1654,"ext": "jpg","anno_path": "test_annotations_first_frame/15.txt","object_class": "vehicle"},
            { "name": "16","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1594,"ext": "jpg","anno_path": "test_annotations_first_frame/16.txt","object_class": "vehicle"},
            { "name": "17","path": "Compressed/test",
                "startFrame": 1411,"endFrame": 1533,"ext": "jpg","anno_path": "test_annotations_first_frame/17.txt","object_class": "vehicle"},
            { "name": "18","path": "Compressed/test",
                "startFrame": 2412,"endFrame": 2697,"ext": "jpg","anno_path": "test_annotations_first_frame/18.txt","object_class": "vehicle"},
            { "name": "19","path": "Compressed/test",
                "startFrame": 2412,"endFrame": 2697,"ext": "jpg","anno_path": "test_annotations_first_frame/19.txt","object_class": "vehicle"},
            { "name": "20","path": "Compressed/test",
                "startFrame": 2412,"endFrame": 2697,"ext": "jpg","anno_path": "test_annotations_first_frame/20.txt","object_class": "vehicle"},
            { "name": "21","path": "Compressed/test",
                "startFrame": 2412,"endFrame": 2697,"ext": "jpg","anno_path": "test_annotations_first_frame/21.txt","object_class": "vehicle"},
            { "name": "22","path": "Compressed/test",
                "startFrame": 2698,"endFrame": 2979,"ext": "jpg","anno_path": "test_annotations_first_frame/22.txt","object_class": "vehicle"},
            { "name": "23","path": "Compressed/test",
                "startFrame": 2698,"endFrame": 3665,"ext": "jpg","anno_path": "test_annotations_first_frame/23.txt","object_class": "vehicle"},
            { "name": "24","path": "Compressed/test",
                "startFrame": 2698,"endFrame": 3168,"ext": "jpg","anno_path": "test_annotations_first_frame/24.txt","object_class": "vehicle"},
            { "name": "25","path": "Compressed/test",
                "startFrame": 2698,"endFrame": 3665,"ext": "jpg","anno_path": "test_annotations_first_frame/25.txt","object_class": "vehicle"},
            { "name": "26","path": "Compressed/test",
                "startFrame": 2698,"endFrame": 3665,"ext": "jpg","anno_path": "test_annotations_first_frame/26.txt","object_class": "vehicle"},
            { "name": "27","path": "Compressed/test",
                "startFrame": 2698,"endFrame": 3665,"ext": "jpg","anno_path": "test_annotations_first_frame/27.txt","object_class": "vehicle"},
            { "name": "28","path": "Compressed/test",
                "startFrame": 2698,"endFrame": 3665,"ext": "jpg","anno_path": "test_annotations_first_frame/28.txt","object_class": "vehicle"},
            { "name": "29","path": "Compressed/test",
                "startFrame": 2712,"endFrame": 3294,"ext": "jpg","anno_path": "test_annotations_first_frame/29.txt","object_class": "vehicle"},
            { "name": "30","path": "Compressed/test",
                "startFrame": 2712,"endFrame": 3078,"ext": "jpg","anno_path": "test_annotations_first_frame/30.txt","object_class": "vehicle"},
            { "name": "31","path": "Compressed/test",
                "startFrame": 3053,"endFrame": 3484,"ext": "jpg","anno_path": "test_annotations_first_frame/31.txt","object_class": "vehicle"},
            { "name": "32","path": "Compressed/test",
                "startFrame": 7156,"endFrame": 9451,"ext": "jpg","anno_path": "test_annotations_first_frame/32.txt","object_class": "vehicle"},
            { "name": "33","path": "Compressed/test",
                "startFrame": 7156,"endFrame": 8717,"ext": "jpg","anno_path": "test_annotations_first_frame/33.txt","object_class": "vehicle"},
            { "name": "34","path": "Compressed/test",
                "startFrame": 7156,"endFrame": 10378,"ext": "jpg","anno_path": "test_annotations_first_frame/34.txt","object_class": "vehicle"},
            { "name": "35","path": "Compressed/test",
                "startFrame": 7156,"endFrame": 9833,"ext": "jpg","anno_path": "test_annotations_first_frame/35.txt","object_class": "vehicle"},
            { "name": "36","path": "Compressed/test",
                "startFrame": 7156,"endFrame": 7291,"ext": "jpg","anno_path": "test_annotations_first_frame/36.txt","object_class": "vehicle"},
            { "name": "37","path": "Compressed/test",
                "startFrame": 7156,"endFrame": 7320,"ext": "jpg","anno_path": "test_annotations_first_frame/37.txt","object_class": "vehicle"},
            { "name": "38","path": "Compressed/test",
                "startFrame": 7416,"endFrame": 9732,"ext": "jpg","anno_path": "test_annotations_first_frame/38.txt","object_class": "vehicle"},
            { "name": "39","path": "Compressed/test",
                "startFrame": 8239,"endFrame": 9625,"ext": "jpg","anno_path": "test_annotations_first_frame/39.txt","object_class": "vehicle"},
            { "name": "40","path": "Compressed/test",
                "startFrame": 9626,"endFrame": 9753,"ext": "jpg","anno_path": "test_annotations_first_frame/40.txt","object_class": "vehicle"},
            { "name": "41","path": "Compressed/test",
                "startFrame": 9733,"endFrame": 10413,"ext": "jpg","anno_path": "test_annotations_first_frame/41.txt","object_class": "vehicle"},
            { "name": "42","path": "Compressed/test",
                "startFrame": 9838,"endFrame": 9970,"ext": "jpg","anno_path": "test_annotations_first_frame/42.txt","object_class": "vehicle"},
            { "name": "43","path": "Compressed/test",
                "startFrame": 9838,"endFrame": 9970,"ext": "jpg","anno_path": "test_annotations_first_frame/43.txt","object_class": "vehicle"},
            { "name": "44","path": "Compressed/test",
                "startFrame": 10039,"endFrame": 10303,"ext": "jpg","anno_path": "test_annotations_first_frame/44.txt","object_class": "vehicle"},
            { "name": "45","path": "Compressed/test",
                "startFrame": 10077,"endFrame": 10560,"ext": "jpg","anno_path": "test_annotations_first_frame/45.txt","object_class": "vehicle"},
            { "name": "46","path": "Compressed/test",
                "startFrame": 10080,"endFrame": 10233,"ext": "jpg","anno_path": "test_annotations_first_frame/46.txt","object_class": "vehicle"},
            { "name": "47","path": "Compressed/test",
                "startFrame": 10146,"endFrame": 11239,"ext": "jpg","anno_path": "test_annotations_first_frame/47.txt","object_class": "vehicle"},
            { "name": "48","path": "Compressed/test",
                "startFrame": 10379,"endFrame": 10524,"ext": "jpg","anno_path": "test_annotations_first_frame/48.txt","object_class": "vehicle"},
            { "name": "49","path": "Compressed/test",
                "startFrame": 10561,"endFrame": 11293,"ext": "jpg","anno_path": "test_annotations_first_frame/49.txt","object_class": "vehicle"},
            { "name": "50","path": "Compressed/test",
                "startFrame": 10612,"endFrame": 11293,"ext": "jpg","anno_path": "test_annotations_first_frame/50.txt","object_class": "vehicle"},
            { "name": "51","path": "Compressed/test",
                "startFrame": 10833,"endFrame": 11293,"ext": "jpg","anno_path": "test_annotations_first_frame/51.txt","object_class": "vehicle"},
            { "name": "52","path": "Compressed/test",
                "startFrame": 10842,"endFrame": 11293,"ext": "jpg","anno_path": "test_annotations_first_frame/52.txt","object_class": "vehicle"},
            { "name": "53","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/53.txt","object_class": "vehicle"},
            { "name": "54","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/54.txt","object_class": "vehicle"},
            { "name": "55","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/55.txt","object_class": "vehicle"},
            { "name": "56","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/56.txt","object_class": "vehicle"},
            { "name": "57","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/57.txt","object_class": "vehicle"},
            { "name": "58","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/58.txt","object_class": "vehicle"},
            { "name": "59","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/59.txt","object_class": "vehicle"},
            { "name": "60","path": "Compressed/test",
                "startFrame": 12009,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/60.txt","object_class": "vehicle"},
            { "name": "61","path": "Compressed/test",
                "startFrame": 12136,"endFrame": 12294,"ext": "jpg","anno_path": "test_annotations_first_frame/61.txt","object_class": "vehicle"},
            { "name": "62","path": "Compressed/test",
                "startFrame": 14393,"endFrame": 14947,"ext": "jpg","anno_path": "test_annotations_first_frame/62.txt","object_class": "vehicle"},
            { "name": "63","path": "Compressed/test",
                "startFrame": 14720,"endFrame": 14941,"ext": "jpg","anno_path": "test_annotations_first_frame/63.txt","object_class": "vehicle"},
            { "name": "64","path": "Compressed/test",
                "startFrame": 15050,"endFrame": 15202,"ext": "jpg","anno_path": "test_annotations_first_frame/64.txt","object_class": "vehicle"},
            { "name": "65","path": "Compressed/test",
                "startFrame": 15241,"endFrame": 15375,"ext": "jpg","anno_path": "test_annotations_first_frame/65.txt","object_class": "vehicle"},
            { "name": "66","path": "Compressed/test",
                "startFrame": 15376,"endFrame": 15735,"ext": "jpg","anno_path": "test_annotations_first_frame/66.txt","object_class": "vehicle"},
            { "name": "67","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22302,"ext": "jpg","anno_path": "test_annotations_first_frame/67.txt","object_class": "vehicle"},
            { "name": "68","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22711,"ext": "jpg","anno_path": "test_annotations_first_frame/68.txt","object_class": "vehicle"},
            { "name": "69","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22375,"ext": "jpg","anno_path": "test_annotations_first_frame/69.txt","object_class": "vehicle"},
            { "name": "70","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22721,"ext": "jpg","anno_path": "test_annotations_first_frame/70.txt","object_class": "vehicle"},
            { "name": "71","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22458,"ext": "jpg","anno_path": "test_annotations_first_frame/71.txt","object_class": "vehicle"},
            { "name": "72","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22406,"ext": "jpg","anno_path": "test_annotations_first_frame/72.txt","object_class": "vehicle"},
            { "name": "73","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22711,"ext": "jpg","anno_path": "test_annotations_first_frame/73.txt","object_class": "vehicle"},
            { "name": "74","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 21682,"ext": "jpg","anno_path": "test_annotations_first_frame/74.txt","object_class": "vehicle"},
            { "name": "75","path": "Compressed/test",
                "startFrame": 21447,"endFrame": 22360,"ext": "jpg","anno_path": "test_annotations_first_frame/75.txt","object_class": "vehicle"},
            { "name": "76","path": "Compressed/test",
                "startFrame": 22376,"endFrame": 22756,"ext": "jpg","anno_path": "test_annotations_first_frame/76.txt","object_class": "vehicle"},
            { "name": "77","path": "Compressed/test",
                "startFrame": 22376,"endFrame": 22595,"ext": "jpg","anno_path": "test_annotations_first_frame/77.txt","object_class": "vehicle"},
            { "name": "78","path": "Compressed/test",
                "startFrame": 22468,"endFrame": 22702,"ext": "jpg","anno_path": "test_annotations_first_frame/78.txt","object_class": "vehicle"},
            { "name": "79","path": "Compressed/test",
                "startFrame": 22610,"endFrame": 22930,"ext": "jpg","anno_path": "test_annotations_first_frame/79.txt","object_class": "vehicle"},
            { "name": "80","path": "Compressed/test",
                "startFrame": 23646,"endFrame": 23931,"ext": "jpg","anno_path": "test_annotations_first_frame/80.txt","object_class": "vehicle"}
        ]

        return sequence_info_list
