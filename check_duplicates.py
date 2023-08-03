import pandas as pd

# Assuming you have already loaded the DataFrame 'bumble_df'

# Print rows with duplicate 'replyContent' values
bumble_df = pd.read_csv('data/bumble_reviews.csv')
duplicates = bumble_df[bumble_df.duplicated('replyContent', keep=False)]
print(duplicates)
