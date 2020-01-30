import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import sys
import os

def transition_states(list_of_paths):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    transition_states = {x + '>' + y: 0 for x in list_of_unique_channels for y in list_of_unique_channels}

    for possible_state in list_of_unique_channels:
        if possible_state not in ['Conversion', 'Null']:
            for user_path in list_of_paths:
                if possible_state in user_path:
                    indices = [i for i, s in enumerate(user_path) if possible_state in s]
                    for col in indices:
                        transition_states[user_path[col] + '>' + user_path[col + 1]] += 1

    return transition_states

def transition_prob(trans_dict, list_of_paths):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    trans_prob = defaultdict(dict)
    for state in list_of_unique_channels:
        if state not in ['Conversion', 'Null']:
            counter = 0
            index = [i for i, s in enumerate(trans_dict) if state + '>' in s]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    counter += trans_dict[list(trans_dict)[col]]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    state_prob = float((trans_dict[list(trans_dict)[col]])) / float(counter)
                    trans_prob[list(trans_dict)[col]] = state_prob

    return trans_prob

def transition_matrix(transition_probabilities, list_of_paths):
    trans_matrix = pd.DataFrame()
    list_of_unique_channels = set(x for element in list_of_paths for x in element)

    for channel in list_of_unique_channels:
        trans_matrix[channel] = 0.00
        trans_matrix.loc[channel] = 0.00
        trans_matrix.loc[channel][channel] = 1.0 if channel in ['Conversion', 'Null'] else 0.0

    for key, value in transition_probabilities.items():
        origin, destination = key.split('>')
        trans_matrix.at[origin, destination] = value

    return trans_matrix

def plotHeatMap(df, dropDuplicates = True):
    # Exclude duplicate correlations by masking upper right values
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, 
                annot=True,
                ax=ax)
    else:
        sns.heatmap(df, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, 
                annot=True,
                ax=ax)

    ax.set_title('Customer journeys through channels')
    plt.show()

def removal_effects(df, conversion_rate):
    removal_effects_dict = {}
    channels = [channel for channel in df.columns if channel not in ['Start',
                                                                     'Null',
                                                                     'Conversion']]
    for channel in channels:
        removal_df = df.drop(channel, axis=1).drop(channel, axis=0)
        for column in removal_df.columns:
            row_sum = np.sum(list(removal_df.loc[column]))
            null_pct = float(1) - row_sum
            if null_pct != 0:
                removal_df.loc[column]['Null'] = null_pct
            removal_df.loc['Null']['Null'] = 1.0

        removal_to_conv = removal_df[
            ['Null', 'Conversion']].drop(['Null', 'Conversion'], axis=0)
        removal_to_non_conv = removal_df.drop(
            ['Null', 'Conversion'], axis=1).drop(['Null', 'Conversion'], axis=0)

        removal_inv_diff = np.linalg.inv(
            np.identity(
                len(removal_to_non_conv.columns)) - np.asarray(removal_to_non_conv))
        removal_dot_prod = np.dot(removal_inv_diff, np.asarray(removal_to_conv))
        removal_cvr = pd.DataFrame(removal_dot_prod,
                                   index=removal_to_conv.index)[[1]].loc['Start'].values[0]
        removal_effect = 1 - removal_cvr / conversion_rate
        removal_effects_dict[channel] = removal_effect

    return removal_effects_dict

def plotRemovalEffects(removal_effects):
    plt.subplots(figsize=(18, 9))    
    keys_list = list(removal_effects.keys())
    values_list = list(removal_effects.values())
    values_list = [100 * v for v in values_list]
    sns.barplot(keys_list, values_list)
    plt.xlabel("Channels")
    plt.ylabel("Conversion loss for channel removal (%)")    
    plt.show()

def markov_chain_allocations(removal_effects, total_conversions):
    re_sum = np.sum(list(removal_effects.values()))

    return {k: (v / re_sum) * total_conversions for k, v in removal_effects.items()}

def plotAttributions(attributions, mean):
    plt.subplots(figsize=(18, 9))
    keys_list = list(attributions.keys())
    values_list = list(attributions.values())
    values_list = [mean * v for v in values_list]
    sns.barplot(keys_list, values_list)
    plt.xlabel("Channels")
    plt.ylabel("Revenue")    
    plt.show()

def main(ca):

    ############################################# Data Preprocessing #############################################

    # Calculate 'mean' of revenue made
    mean = ca['REVENUE'].mean()

    # Since we miss a boolean indicator column for conversion in the dataset, we'll create one
    # Lambda definition: If current 'REVENUE' value is not NaN and its value is greater than 0, that marks a successful transaction
    # so apply True to dataFrame which represents a conversion and False which represesent no conversion
    ca['CONVERSION'] = ca.apply(lambda x: not(math.isnan(x['REVENUE'])) and x['REVENUE'] > 0, axis=1)

    # In order to apply the Markov Chains algorithim (MCA) we need to sort the data by touch-point for each user
    ca = ca.sort_values(['CUSTOMERID', 'TIMESTAMP_TOUCHPOINT'], ascending=[False, True])

    # Add a 'visit order number' to each touch-point for each user by doing cumulative count, but add 1 since 'cumcount()' starts with 0 (just to be real world intuitive)
    ca['VISIT_ORDER'] = ca.groupby('CUSTOMERID').cumcount() + 1

    print(ca[['CUSTOMERID', 'VISIT_ORDER']])
    print('--------------------------------')

    # In order to get the data suitable to apply MCA, we need to shape it so that it contains one row for each user, and the journey they took (paths)
    # for that we convert the data from long-form to wide-form:
    # group dataframe by CUSTOMERID and select the MARKETINGCHANNEL series, after that aggregate the unique MARKETINGCHANNEL's, at the end 
    # reset the index of the DataFrame, and use the default one instead
    ca_paths = ca.groupby('CUSTOMERID')['MARKETINGCHANNEL'].aggregate(lambda x: x.unique().tolist()).reset_index()

    # To add a new 'Null' or 'Conversion' event at the end of the paths (user-journeys) we need to get the last interaction that took place    
    # Note: The dataset is in a sorted state by now, so we keep the last entry for CUSTOMERID.
    ca_last_interaction = ca.drop_duplicates('CUSTOMERID', keep='last')[['CUSTOMERID', 'CONVERSION']]
    
   # Merge the list of final conversion/non-conversion events onto the data frame (left sql join, in order to leave the rows which have no-conversion, so 'Null' for them)
    ca_paths = pd.merge(ca_paths, ca_last_interaction, how='left', on='CUSTOMERID')

    # Construct paths by adding:
    # 1) A 'Start' event at first.
    # 2) User journey through MARKETING CHANNELS (Note these are now sorted by TIMESTAMP_TOUCHPOINT asc).
    # 3) And last depending if a conversion has happend or not, we add 'Null' or 'Conversion' event at the end.
    ca_paths['PATH'] = np.where(ca_paths['CONVERSION'] == False, ['Start'] + ca_paths['MARKETINGCHANNEL'] + ['Null'],
                                                                ['Start'] + ca_paths['MARKETINGCHANNEL'] + ['Conversion'])

    # We only need the CUSTOMERID & PATH Series so we update ca_paths:
    ca_paths = ca_paths[['CUSTOMERID', 'PATH']]

    print(ca_paths)
    print('--------------------------------')

    ################################################ Markov Chains Algorithim ##############################################

    '''
        Summary: 
        
        Markov Chains are very suitable to be applied for channel/revenue attribution since by definition:
            "Markov chains is a stochastic model describing a sequence of possible events in which the probability of each event
             depends only on the state attained in the previous event."

        Which fits perfectly since each customer channel jump, depends only where the user was in the previous channel.

        In order to get the revenue attributions we need to perform 2 steps:
            * Calculate transition probabilities between all states in our state-space [Customer - Path].
            * Calculate removal effects: Remove each channel from the possible paths consecutively and measure how many conversions could be made without the one channel.
    '''

    # Convert ca_paths['PATH'] Series to a list
    list_of_paths = ca_paths['PATH'].tolist()

    # Calculate total number of conversions by iterating over the list of paths, by applying sum() in the result of current path.count('Conversion') on PATH column 
    # Note: path.count('Conversion') will return only where it finds the keyword 'Conversion' on the current value of the PATH column
    total_conversions = sum(path.count('Conversion') for path in list_of_paths)

    # Calculate base conversion rate
    base_conversion_rate = total_conversions / len(list_of_paths)

    # Using touchpoint-timestamps we can calculate transition states (channel jumps), and return a dictionary [transitions, count] for the list of paths.
    # For each unique channel calculate the "jumps" from itself to all other channels
    # Exp: 
    #   * ['SEO_BRAND > SEO_BRAND' (jump on itself), 0]
    #   * ['SEO_BRAND > Direct_NON_BRAND', 3]
    #   ...
    #   * ['SEO_BRAND > Social Media organic_NON-BRAND', 1]
    trans_states = transition_states(list_of_paths)

    # Using above calculated transition states and the frequency of the transitions (given by the count) we can calculate the transition probabilities for the list of paths.
    # Exp:
    #   P(SEO_BRAND, SEO_BRAND) = 0%
    #   P(SEO_BRAND, Direct_NON_BRAND) = 4.58%
    #   P(SEO_BRAND, Social Media organic_NON-BRAND) = 0.28%
    trans_prob = transition_prob(trans_states, list_of_paths)

    # Using above calculated transition probabilities, calculate the transition matrix
    # Exp:
    #   [ P(SEO_BRAND, SEO_BRAND)         P(SEO_BRAND, Direct_NON_BRAND)        .... P(SEO_BRAND, Social Media organic_NON-BRAND)                      ]
    #   [ P(Direct_NON_BRAND, SEO_BRAND)  P(Direct_NON_BRAND, Direct_NON_BRAND) .... P(Direct_NON_BRAND, Social Media organic_NON-BRAND)               ]
    #   [ ...                                                                   .... P(Social Media organic_NON-BRAND, Social Media organic_NON-BRAND) ]
    # Note: Jumps on itself have a 0% probability to happen in our dataset, Why so?
	# Because when we do the basic analysis of the SessionId column we see that it is unique in the dataset. If we had for exmp 2 
	# SessionId’s which where the same and the marketing channel was the same too, that would mean that the user did a channel
	# switch on the same channel, but that is not the case here.
    trans_matrix = transition_matrix(trans_prob, list_of_paths)
    
    # Calculate removal effects
    removal_effects_dict = removal_effects(trans_matrix, base_conversion_rate)
    
    # Find revenue attributions amongst the channels
    attributions_dict = markov_chain_allocations(removal_effects_dict, total_conversions)
    
    ############################################## Data Plotting ###########################################################

    plotHeatMap(trans_matrix, True)
    plotRemovalEffects(removal_effects_dict)
    plotAttributions(attributions_dict, mean)

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    ca = pd.read_csv('{}\Customerattributiondata.tsv'.format(current_dir), delimiter='\t', encoding='utf-8')

    # Basic infos about the dataset
    print(ca.shape)   
    print('--------------------------------')
    print(ca.columns)
    print('--------------------------------')
    print(ca['CUSTOMERID'].describe())
    print('--------------------------------')
    print(ca['SESSIONID'].describe())
    print('--------------------------------')
    print(ca['REVENUE'].describe())
    print('--------------------------------')

    main(ca)
