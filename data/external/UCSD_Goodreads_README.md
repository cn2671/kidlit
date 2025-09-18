# UCSD Goodreads Dataset Information

## Source
- **Dataset**: UCSD Book Graph - Goodreads Dataset
- **URL**: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
- **Collection Date**: Late 2017 (updated May 2019)

## Dataset Description
The UCSD Book Graph contains comprehensive Goodreads data including:
- 2,360,655 books (1,521,962 works, 400,390 book series, 829,529 authors)
- 876,145 users
- 228,648,342 user-book interactions
- 112,131,203 reads and 104,551,549 ratings

## Downloaded Files
- `goodreads_books.json.gz`: Main book metadata (~2GB)
- `goodreads_book_authors.json.gz`: Author information
- `goodreads_book_genres_initial.json.gz`: Genre mappings
- `goodreads_children_sample.json`: Extracted children's books sample

## Usage Restrictions
- **Academic use only**
- Do not redistribute or use for commercial purposes
- Must cite required papers when using

## Citation Requirements
1. "Item Recommendation on Monotonic Behavior Chains" (RecSys'18)
2. "Fine-Grained Spoiler Detection from Large-Scale Review Corpora" (ACL'19)

## Integration with KidLit Project
This dataset can be used for:
- Cross-referencing book metadata and ratings
- Feature engineering for recommendation system
- Validation of Lexile estimates
- Enhanced book discovery and recommendation

## Next Steps
1. Process children's book sample for relevant features
2. Match books with existing KidLit dataset
3. Extract additional metadata (genres, ratings, reviews)
4. Use for model validation and enhancement