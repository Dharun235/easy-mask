from easy_mask.io.images import image_size, list_images


def test_list_images_natural_order_and_filters(image_dir):
    assert [p.name for p in list_images(image_dir)] == ["img1.png", "img2.png", "img10.png"]


def test_image_size(image_dir):
    assert image_size(image_dir / "img1.png") == (100, 80)
