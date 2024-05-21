#!/usr/bin/env python
# coding: utf-8
# # Step 1: Collect, Select, Sort, and Cleanup the Sefaria Dataset
# 
# "Shas" here is going to include all four types of mishnaic literature: Mishnah, Tosefta, Talmud Bavli, and Talmud Yerushalmi - and we don't care which one is referenced, but we don't want to count them all more than once. So I'll just collect all those references and note which "tractate" of the 63 tractates they belong to.
# 
# Two tiny bits of cleanup are also required:
# 1. Because Sefaria's dataset only includes links in one direction, we will duplicate those so that every cross-reference is bidirectional (i.e., for every time Ketubot quotes Chullin, that means Chullin also quotes Ketubot)
# 1. "Ohalot" and "Uktzin" have two different spellings for their Tosefta name (go figure)

import requests
import json, urllib.request
import pandas as pd

url = 'https://raw.githubusercontent.com/Sefaria/Sefaria-Export/master/links/links_by_book_without_commentary.csv'
df = pd.read_csv(url)

BTtractates = [ #checked that this lists accords with Sefaria title transliterations
    "Berakhot", "Shabbat", "Eruvin", "Pesachim", "Yoma", "Sukkah", "Beitzah", "Rosh Hashanah", "Taanit", "Megillah", "Moed Katan", "Chagigah",
    "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
    "Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot", "Shevuot", "Avodah Zarah", "Horayot",
    "Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah", "Tamid",
    "Niddah"
]

plainshaslist = [ #this is the names of the 63 masechtos, whether mishnah or other
    'Berakhot', "Peah", 'Demai', "Kilayim", "Sheviit", 'Terumot', "Maasrot", "Maaser Sheni",
    'Challah', 'Orlah', 'Bikkurim', 'Shabbat', 'Eruvin', 'Pesachim', 'Shekalim', 'Yoma', 'Sukkah',
    'Beitzah', 'Rosh Hashanah', 'Taanit', 'Megillah', 'Moed Katan', 'Chagigah', 'Yevamot', 'Ketubot',
    'Nedarim', 'Nazir', 'Sotah', 'Gittin', 'Kiddushin', 'Bava Kamma', 'Bava Metzia', 'Bava Batra',
    'Sanhedrin', 'Makkot', "Shevuot", 'Eduyot', 'Avodah Zarah', 'Pirkei Avot', 'Horayot', 'Zevachim',
    'Menachot', 'Chullin', 'Bekhorot', 'Arakhin', 'Temurah', 'Keritot', "Meilah", 'Tamid', 'Middot',
    'Kinnim', 'Keilim', 'Kelim', 'Ohalot', 'Oholot', "Negaim", 'Parah', 'Tahorot', "Mikvaot", 'Niddah', 'Makhshirin',
    'Zavim', 'Tevul Yom', 'Yadayim', 'Oktzin', 'Oktsin' #yes there are two Ohalot's and two Oktzin's which we will have to merge later
]

shaslist = []

#Allowing for references to Mishnah, Tosefta, and Jerusalem Talmud
for n in plainshaslist:
    mish = "Mishnah " + n
    tos = "Tosefta " + n #We are actaully going to be missing Tosefta Keilim Kama, etc. but I think that's ok
    jeru = "Jerusalem Talmud " + n
    nlist = [mish, tos, jeru, n]
    shaslist += nlist

# Collect only cross-references of Shas, not commentaries, Tanakh, etc.
filtered1_df = df[df['Text 1'].isin(BTtractates)]
filtered2_df = filtered1_df[filtered1_df['Text 2'].isin(shaslist)]

# Add reverse cross-references (i.e. if Berakhot quotes Shabbat 54 times, make sure that
    # the number 54 is reflected both in 'Berakhot > Shabbat' and in 'Shabbat > Berakhot')
# Create an empty DataFrame to collect new rows that need to be added
new_rows = []

# Iterate through the DataFrame to find rows to reverse
for index, row in filtered2_df.iterrows():
    if row['Text 2'] in BTtractates:
        # Create a new row with "Text 1" and "Text 2" reversed, while keeping "Tractate" and "Link Count" the same
        new_row = pd.DataFrame({
            'Text 1': [row['Text 2']],
            'Text 2': [row['Text 1']],
            'Link Count': [row['Link Count']]
        })
        new_rows.append(new_row)

# Concatenate the new rows with the original DataFrame
df_augmented = pd.concat([filtered2_df] + new_rows, ignore_index=True)

# create a sorting scheme
df_sort = df_augmented.copy()  # Making a copy to avoid affecting the original df
df_sort['Text 1'] = pd.Categorical(df_sort['Text 1'], categories=BTtractates, ordered=True)
sorted_df = df_sort.sort_values('Text 1')

crossrefs = sorted_df #making another copy

# Convert "Text 2" to a categorical type with categories in the specified order of shaslist
crossrefs['Text 2'] = pd.Categorical(crossrefs['Text 2'], categories=shaslist, ordered=True)

# Now sort the DataFrame first by "Text 1" and then by "Text 2"
crossrefs_sorted = crossrefs.sort_values(['Text 1', 'Text 2'])
crossrefs_sorted


# Now let's find the greatest number of links for each "tractate," including Mishnah/Tosefta/Yerushalmi/Bavli

crossrefs_copy = crossrefs_sorted #copying again just in case

#We have to fix "Oktzin" and "Ohalot":
crossrefs_copy = crossrefs_copy.replace(to_replace='Oktsin', value='Oktzin', regex=True)
crossrefs_copy = crossrefs_copy.replace(to_replace='Oholot', value='Ohalot', regex=True)

# Function to collapse all the different types of "Tractates" by removing prefixes
def remove_prefixes(text):
    # Define prefixes to remove
    prefixes = ["Mishnah ", "Tosefta ", "Jerusalem Talmud "]
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]  # Remove the prefix and return
    return text  # Return the original text if no prefix matched

# Create a new "Tractate" column that will apply the function
crossrefs_copy['Tractate'] = crossrefs_copy['Text 2'].apply(remove_prefixes)

#Re-sort
# create a sorting scheme
df_sort = crossrefs_copy.copy()  # Making a copy to avoid affecting the original df
df_sort['Text 1'] = pd.Categorical(df_sort['Text 1'], categories=BTtractates, ordered=True)
sorted_df = df_sort.sort_values('Text 1')
sorted_df['Text 2'] = pd.Categorical(sorted_df['Text 2'], categories=shaslist, ordered=True)

# Now sort the DataFrame first by "Text 1" and then by "Text 2"
crossrefs_copy_sorted = crossrefs_copy.sort_values(['Text 1', 'Text 2'])

# reset index values to go in order
crossrefs_copy_sorted = crossrefs_copy_sorted.reset_index(drop=True)


crossrefs_copy_sorted.to_csv('all-references.csv', sep=',')

# # Approach 1: Count the Number of Cross-References
# We can't just sum up all the references to Mishnah, Tosefta, Talmud Yerushalmi and Bavli, because many of those will be duplicates (for example, a link might exist to a Mishnah which also appears in the Tosefta and Talmuds, and give a score of 4, when really it should just be counted as 1). So we'll just count the *greatest* numerical value from among any of the possible references within a single tractate. And another small bit of cleaning up to remove self-referenes (Berakhot to Berakhot, whether Tosefta, Mishnah, etc.)

# Remove all rows where 'Text 1' is 'Text 2'
crossrefs_copy_sorted['Text 1'] = crossrefs_copy_sorted['Text 1'].astype(str) #bit of a roundabout way but otherwise python doesn't recognize the values as identical
crossrefs_copy_sorted['Text 2'] = crossrefs_copy_sorted['Text 2'].astype(str)
crossrefs_copy_sorted = crossrefs_copy_sorted[crossrefs_copy_sorted['Text 1'] != crossrefs_copy_sorted['Text 2']]

# Create a new dataset for "Link Count values"
# Initialize an empty dictionary to hold the maximum "Link Count" for each unique pair of "Text 1" and "Tractate"
max_link_dict = {}

# Iterate over each row in the DataFrame to populate the dictionary
for _, row in crossrefs_copy_sorted.iterrows():
    key = (row['Text 1'], row['Tractate'])  # Unique key is a tuple of "Text 1" and "Tractate"
    link_count = row['Link Count']
    text2 = row['Text 2']

    # If the key is not in the dictionary or the current row's "Link Count" is greater than the stored value, update the dictionary
    if key not in max_link_dict or link_count > max_link_dict[key]['Link Count']:
        max_link_dict[key] = {'Text 2': text2, 'Link Count': link_count}

# Convert the dictionary back to a DataFrame for easy viewing and manipulation
max_link_df = pd.DataFrame([(*key, value['Text 2'], value['Link Count']) for key, value in max_link_dict.items()], columns=['Text 1', 'Tractate', 'Text 2', 'Link Count'])

# Now again remove all rows where 'Text 1' is equivalent to 'Tractate'
max_link_df['Text 1'] = max_link_df['Text 1'].astype(str)
max_link_df['Tractate'] = max_link_df['Tractate'].astype(str)
max_link_df = max_link_df[max_link_df['Text 1'] != max_link_df['Tractate']]

# Create another new dataset named 'score' to count the number of unique 'Tractate' values associated with each 'Text 1' value

# Group the DataFrame by 'Text 1' and then count the number of unique 'Tractate' values for each group
score = max_link_df.groupby('Text 1')['Tractate'].nunique().reset_index()

# Rename columns for clarity
score.columns = ['Text 1', 'Unique Tractate Count']

score_sorted = score.sort_values(by='Unique Tractate Count', ascending=False)

# reset index values to go in order
score_sorted = score_sorted.reset_index(drop=True)


#  In theory, I believe would be the most accurate reflection of 'shas-kattan-ness' would be to score each tractate's references as follows: give one point for every unique Tractate referenced. For the second time that tractate is referenced, give 0.5 points, then the third time, give 0.25, etc. (So, for example, if a tractate quotes Berakhot once and Shabbat thrice, it will have a score of 1 + 1.75 = 2.75). Below we score the cross-references in such a way, but perhaps a more exponential discounting would be better, I dunno

#Now for this more complicated scoring technique using a geometrically decreasing for each additional link per tractate
data = max_link_df

# Calculate the "Link Count Score" for each row
data['Link Count Score'] = data['Link Count'].apply(lambda n: (1 - (0.5 ** n)) / (1 - 0.5))

# Group by "Text 1" and sum the "Link Count Score" for each group
final_scores = data.groupby('Text 1')['Link Count Score'].sum().reset_index()

# Now combine them all and view the dataset
combined_df = pd.merge(score_sorted, final_scores, left_on='Text 1', right_on='Text 1', how='left')

# Also renaming columns to something understandable
combined_df.rename(columns={'Text 1': 'Tractate', 'Unique Tractate Count': 'Unique Tractates Referenced', 'Link Count Score': 'Exponential-Decrease-Score'}, inplace=True)
combined_df

### Make pretty pictures
df = combined_df[['Tractate', 'Unique Tractates Referenced', 'Exponential-Decrease-Score']]
df = df.sort_values(by=['Unique Tractates Referenced', 'Exponential-Decrease-Score'], ascending=False)

import matplotlib.pyplot as plt
import networkx as nx

# Renaming columns according to chord diagram convention
df.columns = ['Source', 'Target', 'Value']

# Define the sedarim and their tractates for color-coding the nodes
sedarim = {
    'Zeraim': ['Berakhot', "Peah", 'Demai', "Kilayim", "Sheviit", 'Terumot', "Maasrot", "Maaser Sheni", "Challah", 'Orlah', 'Bikkurim'],
    'Moed': ['Shabbat', 'Eruvin', 'Pesachim', 'Shekalim', 'Yoma', 'Sukkah','Beitzah', 'Rosh Hashanah', 'Taanit', 'Megillah', 'Moed Katan', 'Chagigah'],
    'Nashim': ['Yevamot', 'Ketubot','Nedarim', 'Nazir', 'Sotah', 'Gittin', 'Kiddushin'],
    'Nezikin': ['Bava Kamma', 'Bava Metzia', 'Bava Batra','Sanhedrin', 'Makkot', "Shevuot", 'Eduyot', 'Avodah Zarah', 'Pirkei Avot', 'Horayot'],
    'Kodshim': ['Zevachim', 'Menachot', 'Chullin', 'Bekhorot', 'Arakhin', 'Temurah', 'Keritot', "Meilah", 'Tamid', 'Middot','Kinnim'],
    'Taharot': ['Keilim', 'Kelim', 'Ohalot', 'Oholot', "Negaim", 'Parah', 'Tahorot', "Mikvaot", 'Niddah', 'Makhshirin', 'Zavim', 'Tevul Yom', 'Yadayim', 'Oktzin', 'Oktsin']
}

# Map each tractate to its Seder
tractate_to_seder = {tractate: seder for seder, tractates in sedarim.items() for tractate in tractates}

# Define a color map for each of the sedarim
seder_colors = {
    'Zeraim': 'red',
    'Moed': 'blue',
    'Nashim': 'green',
    'Nezikin': 'yellow',
    'Kodshim': 'purple',
    'Taharot': 'orange'
}
# Create a graph from the dataframe
G = nx.from_pandas_edgelist(df, 'Source', 'Target', ['Value'])

# Transfer the color list to nodes
default_color = 'grey' # just in case there's something not in the list (i.e. misspelled, etc), I'll find it
node_colors = [seder_colors.get(tractate_to_seder.get(node, None), default_color) for node in G.nodes()]

# Draw the network
plt.figure(figsize=(18, 18))
pos = nx.spring_layout(G, seed=42)  # for consistent layout
nx.draw_networkx_nodes(G, pos, node_size=2, node_color=node_colors, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=12)
#edges = nx.draw_networkx_edges(G, pos)
plt.axis('off')
plt.show()


# # Approach 2: Balance Between "Six Orders"
# There is another possible way of interpreting "shas kattan"-ness, which would refer to how well all of the citations are in terms of being a fairer representation of all of the six orders of the Mishnah. A perfectly 'balanced' tractate will have 1/6 of its references to Zera'im, 1/6 to Mo'ed, etc. (although the six orders themselves are not all the same size or have the same number of tractates, so perhaps it makes sense to adjust for that). If we consider each inter-talmudic reference according to one of the six orders, which tractate is closest to the 'balanced ideal' of 1/6 of its references alloted to each order? Here too I just used the cross-references (“link”) count from Sefaria’s github, and just categorized the results based on Seder which are color-coded differently in the bar graph below (click here for a colorblind reader friendly version). Even without calculating a “balance score,” you can easily see that the tractate of Talmud Bavli with the most evenly balanced set of references is Niddah. 


# Starting again with the full dataset of references
data = crossrefs_copy_sorted

# Define the sedarim and their tractates for mapping
sedarim = {
    'Zeraim': ['Berakhot', "Peah", 'Demai', "Kilayim", "Sheviit", 'Terumot', "Maasrot", "Maaser Sheni", "Challah", 'Orlah', 'Bikkurim'],
    'Moed': ['Shabbat', 'Eruvin', 'Pesachim', 'Shekalim', 'Yoma', 'Sukkah','Beitzah', 'Rosh Hashanah', 'Taanit', 'Megillah', 'Moed Katan', 'Chagigah'],
    'Nashim': ['Yevamot', 'Ketubot','Nedarim', 'Nazir', 'Sotah', 'Gittin', 'Kiddushin'],
    'Nezikin': ['Bava Kamma', 'Bava Metzia', 'Bava Batra','Sanhedrin', 'Makkot', "Shevuot", 'Eduyot', 'Avodah Zarah', 'Pirkei Avot', 'Horayot'],
    'Kodshim': ['Zevachim', 'Menachot', 'Chullin', 'Bekhorot', 'Arakhin', 'Temurah', 'Keritot', "Meilah", 'Tamid', 'Middot','Kinnim'],
    'Taharot': ['Keilim', 'Kelim', 'Ohalot', 'Oholot', "Negaim", 'Parah', 'Tahorot', "Mikvaot", 'Niddah', 'Makhshirin', 'Zavim', 'Tevul Yom', 'Yadayim', 'Oktzin', 'Oktsin']
}

# Map each tractate to its Seder
tractate_to_seder = {tractate: seder for seder, tractates in sedarim.items() for tractate in tractates}

# Add the 'Seder' column to the dataset
data['Seder'] = data['Tractate'].map(tractate_to_seder)
data.to_csv('citation_counts.csv', sep=',') # Saving this dataset

## Now sum up/count the citations for each Seder
# Summing up the link counts for each combination of tractate ('Text 1') and order ('Seder')
grouped_data = data.groupby(['Text 1', 'Seder'])['Link Count'].sum().unstack(fill_value=0)

# Calculating the total number of citations for each tractate
total_citations = grouped_data.sum(axis=1)
grouped_data.to_csv('seder_counts.csv', sep=',') #Note that this dataset put the Seder counts in alphabetical order (see 'category_names' below)

## Use the saved dataset to make a nice boxplot
# Converting the saved data back to a dictionary 

# Define category names and order
category_names = ['Kodshim', 'Moed', 'Nashim', 'Nezikin', 'Taharot', 'Zeraim']
new_order = ['Zeraim', 'Moed', 'Nashim', 'Nezikin', 'Kodshim', 'Taharot']
category_colors = plt.get_cmap('tab20c')(np.linspace(0, 1, len(new_order)))

# Normalize the data
data_normalized = data[category_names].div(data[category_names].sum(axis=1), axis=0)

# Reorder data
data_normalized = data_normalized[new_order]

# Plot the figure
fig, ax = plt.subplots(figsize=(10, 18), dpi=300)
ax.invert_yaxis()
ax.set_xlim(0, 1)

for i, category in enumerate(new_order):
    widths = data_normalized[category]
    starts = data_normalized[new_order[:i]].sum(axis=1)
    ax.barh(data['Text 1'], widths, left=starts, height=0.5, color=category_colors[i], label=category)

# Adjust the legend
ax.legend(title="Seder", bbox_to_anchor=(0.5, 1.15), loc='upper center', fontsize='small', ncol=len(new_order))

# Save the figure
plt.savefig('order_balance.png', bbox_inches='tight')


### Approach 3: Using Sefaria's "Topics" Ontology Tool

#downlaod the giant index list
import requests

url = "https://www.sefaria.org/api/index/"
response = requests.get(url)
data = response.json()

# A kinda roundabout way of collecting all the titles of the Talmud Bavli, according to Sefaria's API
# Requires a few levels of parsing the json object 
data_list = list(data) 
fullmetadict = dict(data_list[2])
fulldict = fullmetadict['contents']
fulldict = fulldict[0] #this contains all of the "Talmud" corpus

whatlist = []

#now I'll collect the next level down (which includes Zera'im, Mo'ed, but also Commentaries, etc.)
for k,v in fulldict.items():
    if k == 'contents':
        for i in v:
            for new_i in v:
                whatlist.append(new_i)

bavli_titles = []

# Now I can collect only the Talmud Bavli titles, which are contained in the first 6 Talmud categories
for i in whatlist[:6]:
    sederdict = i
    for v in sederdict['contents']:
        bavli_titles.append(v['title'])

# Function for using the Talmud Bavli name to get all the related topics
def get_topics_for_book(book_name):
    # Build the text reference
    text_ref = f"{book_name}"

    # Endpoint URL for getting topics associated with a text
    url = f"https://www.sefaria.org/api/related/{text_ref}"

    # Make the GET request
    response = requests.get(url, timeout = 600)
    if response.status_code == 200: #error handling
        topics = response.json()
        return topics
    else:
        return f"Failed to retrieve topics: Status Code {response.status_code}"


# This will collect a names of all the unique topics associated with each tractate
bavli_topics_lists = {}
bavli_topics_num = {} #and this will keep the count

for book in bavli_titles:
    try:
        links = get_topics_for_book(book) #See function
    
        topicsdetails = links['topics'] # Parse json by collecting the topics 

        # Make a list of each topic that is connected to this book...
        topicnames = []

        # By parsing through the dictionary object of all the topics 
        for i in topicsdetails:
            topicnames.append(i['topic']) #and just collect the actual name of the topic
    
        #remove duplicates from the topic list
        topicnames = list(set(topicnames))

        # Assign the topic list, and its length, to dictionaries with the tractate name as the key
        bavli_topics_lists[book] = topicnames 
        bavli_topics_num[book] = len(topicnames)
        #print('successfully analyzed {book}!')

    except:
        print(f'failed to collect data for {book})

# Now let's see which tractates have the most associated topics 
# Convert the dictionary into a dataframe
topicsdf = pd.DataFrame([(key, value) for key, value in bavli_topics_num.items()], columns=['Tractate', 'Topics Count'])

# create a sorting scheme
topicsdf = topicsdf.sort_values('Topics Count', ascending=False)
topicsdf

