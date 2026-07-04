Step 1: Read the Dataset

Input : data/raw/sherlock.txt
Goal  : Load the complete book into memory.

Step 2: Clean the Text

1. Remove Project Gutenberg Header/Footer
2. Normalize Whitespace
3. Lowercase ? Convert everything to lowercase
4. Remove extra spaces
5. Punctuation?  For Word2Vec v1: I recommend removing punctuation. Later we'll build a tokenizer that keeps punctuation as tokens.

Step 3: Tokenization

Now split text into words.

Step 4: Build Vocabulary

Step 5: Convert Tokens to IDs

Neural networks only understand numbers.


Step 6: Save the Artifacts

Instead of rebuilding the vocabulary every run, save it.

