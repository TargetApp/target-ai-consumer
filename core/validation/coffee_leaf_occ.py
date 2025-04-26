from .. import ModelWrapper


class CoffeeLeafOCC(ModelWrapper):
    """Wrapper for one-class classification models that identify if an image contains a coffee leaf."""

    def __call__(self, image: bytes, *args, **kwargs) -> tuple[int]:
        """Classify an image.

        Parameters
        ----------
        image : bytes
            The image to classify.

        Returns
        -------
        tuple[int]
            The classification result.
        """
        image = self.load_image(image)
        out_is_coffee_leaf = self.model(image)
        is_coffee_leaf = out_is_coffee_leaf.argmax().item() ^ 1
        return (is_coffee_leaf,)
