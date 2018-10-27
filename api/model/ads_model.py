from api.model.base_model import BaseModel


class AdsModel(BaseModel):

    def predict(self, words_raw):
        """Returns list of tags
        Args:
            words_raw: list of words (string), just one sentence (no batch)
        Returns:
            preds: list of tags (string), one for each word in the sentence
        """
        return [0]