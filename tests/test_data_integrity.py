import hashlib
import json
import os
import unittest

import _path  # noqa: F401
from PIL import Image

import common as C

EXPECT = {  # filename -> (size, ISO, exposure seconds)
    "IMG_20261026_195540763.jpg": ((4080, 3072), 734, 0.07),
    "IMG_20261026_195556198.jpg": ((4080, 3072), 6400, 0.06),
    "IMG_20261026_195635983.jpg": ((4080, 3072), 491, 0.07),
    "IMG_20261026_195702582.jpg": ((4080, 3072), 722, 0.02),
}


class TestData(unittest.TestCase):
    def test_photos_present_dimensions_exif(self):
        for f, (size, iso, expo) in EXPECT.items():
            p = os.path.join(C.PHOTOS, f)
            self.assertTrue(os.path.isfile(p), f)
            with Image.open(p) as im:
                self.assertEqual(im.size, size)
                ex = im.getexif().get_ifd(0x8769)
            self.assertEqual(ex.get(0x8827), iso)
            self.assertAlmostEqual(float(ex.get(0x829A)), expo, delta=0.005)

    def test_photo_hashes_match_recorded_originals(self):
        man = os.path.join(C.DATA, "photos_original_sha256.json")
        self.assertTrue(os.path.isfile(man))
        with open(man) as fh:
            ref = json.load(fh)
        for f in EXPECT:
            with open(os.path.join(C.PHOTOS, f), "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()
            self.assertEqual(h, ref[f], f)

    def test_provided_images(self):
        exp = {"q1_testimage.png": (512, 512), "q4_groundtruth.png": (512, 512), "cameraman.tif": (512, 512),
               "characters_test_pattern.tif": (688, 688)}
        for f, shp in exp.items():
            self.assertEqual(C.load_gray(os.path.join(C.PROVIDED, f)).shape, shp)
        with Image.open(os.path.join(C.PROVIDED, "reference_image.jpg")) as ref:
            self.assertGreaterEqual(min(ref.size), 1024)


if __name__ == "__main__":
    unittest.main()
