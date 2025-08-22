# test_smoke.py
from scripts.parse_query import parse_user_query
from scripts.recommender import recommend_books

def main():
    user_input = "I want a fun adventure story for my 7 year old who loves animals."
    print("User says:", user_input)

    filters = parse_user_query(user_input)
    print("Parsed filters:", filters)

    recs = recommend_books(
        age_range=filters.get("age_range"),
        themes=filters.get("themes"),
        tone=filters.get("tone"),
        n=5
    )
    print("\nRecommendations:")
    print(recs.to_string(index=False))

if __name__ == "__main__":
    main()
