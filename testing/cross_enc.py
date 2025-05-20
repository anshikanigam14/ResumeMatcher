from sentence_transformers import CrossEncoder

# Load CrossEncoder model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Query
query = "I want a name to select birds which can fly highest"

# Candidate names with neutral descriptions (no explicit animal/bird words)
candidates = [
    "Lion: Known for strength, courage, and a commanding presence",         # Animal - Lion
    "Tiger: Associated with agility, stealth, and raw power",              # Animal - Tiger
    "Zebra: Recognized for unique black and white patterns and herd life", # Animal - Zebra
    "Sparrow: Small and agile, often seen flitting around gardens",        # Bird
    "Eagle: Soars high with sharp vision and dominance from above",        # Bird
    "Peacock: Displays vibrant colors and is known for its elegance"       # Bird
]

# Pair each candidate with the query
input_pairs = [(query, name_desc) for name_desc in candidates]

# Compute scores
scores = model.predict(input_pairs)

# Sort candidates by score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

# Display top 2
print("Top 2 matches for the query:")
for i, (name_desc, score) in enumerate(ranked, 1):
    print(f"{i}. {name_desc} (Score: {score:.3f})")