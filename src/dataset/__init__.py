from src.dataset.goodreads import GoodreadsDataset
from src.dataset.dblp import DBLPDataset


load_dataset = {
    'goodreads': GoodreadsDataset,
    'dblp': DBLPDataset
}
