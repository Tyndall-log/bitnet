from pathlib import Path

from datasets import load_dataset

root_dir = Path(__file__).parent

ds = load_dataset(
	"BCCard/BCCard-Finance-Kor-QnA",
	cache_dir=(root_dir / "data" / "hf_cache" / "datasets").as_posix(),
)

pass

