import pandas as pd
from collections import defaultdict
import numpy as np


def main(csv_file):
    """
    Given an appropriate csv_file, output the relevant columns.

    Returns df with columns [user, start, end, normalized_company, industry]
    """

    employ_data = pd.read_csv(csv_file, sep="\t", header=None,
                              names=[i for i in range(34)], low_memory=False)
    # column info from taxonomy file
    name = ['user', 'name', 'birth', 'gender', 'primary',
            'primary_weight', 'secondary', 'secondary_weight',
            'city', 'country', 'education', 'elite', 'start',
            '.', 'end', '??', '/', 'length', 'role', 'department',
            'company', 'normalized_company', 'ticker', 'exchange',
            'public', 'location_company', 'industry', 'educational',
            'degree', 'elite_education', 'major', 'department', 'FIGI',
            'last_update']
    #     drop = ['length', 'gender', 'primary',
    #         'primary_weight', 'secondary', 'secondary_weight',
    #         'city', 'country', 'education', 'elite', '.', '??',
    #         '/', 'department', 'exchange',
    #         'public', 'location_company', 'educational', 'degree', 'elite_education',
    #         'major', 'department', 'FIGI', 'last_update']
    employ_data.columns = name
    return employ_data


# data without datetime features, and none values for some dates
raw_data = {'db': main('./Data/DB_profiles.csv'),
            'gs': main('./Data/GS_profiles.csv'),
            'leh': main('./Data/LEH_profiles.csv'),
            'ms': main('./Data/MS_profiles.csv'),
            'ubs': main('./Data/UBS_profiles.csv')
            }


def get_users(company_name, company_data, worked_date='2008-01-01', missing_start='1900-01-01',
              missing_end='2018-01-01'):
    """"
    Returns the users who worked at a given company on worked_date, that does not have both start and
    end dates missing

    worked_date: string specifying the date on which to extract employees from.
                 Must be coercible into a datetime object
    missing_start: default value for missing start dates
    missing_end: default value for missing end dates
    """
    worked_date = pd.to_datetime(worked_date)
    missing_start = pd.to_datetime(missing_start)
    missing_end = pd.to_datetime(missing_end)
    x = company_data

    company_tickers = {'db': 'DB', 'leh': 'LEH', 'gs': 'GS', 'ms': 'MS^E', 'ubs': 'UBS'}
    # conditions: start and end not both missing, worked before/after 2008-01-01, ticker matches company
    mask = ~((x['start'] == missing_start) & (x['end'] == missing_end)) & \
           (x['start'] < worked_date) & \
           (x['end'] > worked_date) & \
           (x['ticker'] == company_tickers[company_name])
    return company_data[mask]['user'].unique()


# gets the user_ids within each company that match the conditioning, before and and after
#   2008-01-01, exclusive
users = {company_name: get_users(company_name, company_data) for company_name, company_data in raw_data.items()}

data = {}
for company, company_data in raw_data.items():
    company_users = users[company]
    data[company] = company_data[company_data['user'].isin(company_users)]

# begin code for filling in missing industries
# read all the csv files
profile = pd.read_csv('./Data/profile_industry_mappings.csv', header=None, names=[i for i in range(5)], dtype={4: str})
profile.drop([0, 2], axis='columns', inplace=True)
profile.rename(mapper={1: 'company', 3: 'norm', 4: "ind"}, axis='columns', inplace=True)

mturk = pd.read_csv('./Data/industries_MTurkers_20170711.csv', header=None, encoding='latin-1')
mturk.drop([1], axis='columns', inplace=True)
mturk.rename(mapper={0: 'company', 2: "ind"}, axis='columns', inplace=True)

finance = pd.read_csv('./Data/Finance.csv', dtype={'Industry': str})
finance.drop([finance.columns[0], finance.columns[2], finance.columns[4]], axis='columns', inplace=True)
finance.rename(mapper={'Normalized Company Name': 'norm', 'Industry': "ind"}, axis='columns', inplace=True)

manual = pd.read_csv('./Data/manual_industry_mappings.csv', encoding='latin-1', header=None, dtype={2: str})
manual.drop([1], axis='columns', inplace=True)
manual.rename(mapper={0: 'norm', 2: "ind"}, axis='columns', inplace=True)

industries_2019 = pd.read_csv('./Data/missing_industries_2019.csv', header=None, dtype={2: str})
industries_2019 = industries_2019[~(industries_2019[1] == 1)].copy()

industries_2019.drop([1], axis='columns', inplace=True)
industries_2019.rename(mapper={0: 'company', 2: "ind"}, axis='columns', inplace=True)
industries_2019 = industries_2019[~pd.isnull(industries_2019.ind)].copy()

# mturk industry is given as "ind_x", profile industry is given as "ind_y"
company_comb = pd.merge(mturk, profile, on='company', how='outer')
# prioritize mturk data
company_comb['combined'] = company_comb['ind_x'].combine_first(company_comb['ind_y'])

# mturk industry is given as "ind", profile industry is given as "combined"
company_comb = pd.merge(industries_2019, company_comb, on='company', how='outer')
# prioritize manual entry data
company_comb['combined'] = company_comb['ind'].combine_first(company_comb['combined'])

# merge manual and finance files, prioritizing manual
norm_comb = pd.merge(manual, finance, on='norm', how='outer')
norm_comb['combined'] = norm_comb['ind_x'].combine_first(norm_comb['ind_y'])
# merge manual/finance and profile[norm], prioritizing manual/finance
norm_comb = pd.merge(norm_comb, profile, on='norm', how='outer')
norm_comb['combined'] = norm_comb['combined'].combine_first(norm_comb['ind'])

# convert the columns of the aggredated dataframe into a dictionary where the key is the company name
# and the value is the industry code
norm_mapping = dict(zip(norm_comb.norm, norm_comb.combined))
company_mapping = dict(zip(company_comb.company, company_comb.combined))
# set the default value if the company is not found to NaN
norm_mapping = defaultdict(lambda: np.NaN, norm_mapping)
company_mapping = defaultdict(lambda: np.NaN, company_mapping)

def filter_manual(company_data):
    """
    Adds industry labels to entries that don't have one, based on the industry data in company_mapping and
    norm_mapping
    """
    company_data = company_data.copy()
    # convert to lowercase for more accurate matching
    company_data['normalized_company_lower'] = company_data['normalized_company'].str.lower()
    company_data['company_lower'] = company_data['company'].str.lower()
    # apply norm_mapping and company_mapping to upper and lower case versions
    company_data['company_mapped'] = company_data['company'].apply(lambda y: company_mapping[y])
    company_data['normalized_company_mapped'] = company_data['normalized_company'].apply(lambda y: norm_mapping[y])
    company_data['company_lower_mapped'] = company_data['normalized_company_lower'].apply(lambda y: norm_mapping[y])
    company_data['normalized_company_lower_mapped'] = company_data['normalized_company_lower'].apply(lambda y: norm_mapping[y])
    # combines all mappings. Prioritize Existing industry code > MTurk/profle(company) >
    # manual/finance/profile(normalized_company) > manual/finance/profile(normalized_company_lower)
    company_data['industry_two'] = company_data['industry'].combine_first(company_data['company_mapped'])
    company_data['industry_three'] = company_data['industry_two'].combine_first(company_data['normalized_company_mapped'])
    company_data['industry_four'] = company_data['industry_three'].combine_first(company_data['company_lower_mapped'])
    company_data['industry_five'] = company_data['industry_four'].combine_first(company_data['normalized_company_lower_mapped'])
    company_data['industry'] = company_data['industry_five']
    # drop the temporary columns
    company_data.drop(['normalized_company_lower', 'company_lower', 'company_mapped', 'normalized_company_mapped', 'company_lower_mapped','normalized_company_lower_mapped', 'industry_two', 'industry_three', 'industry_four','industry_five'], axis=1, inplace=True)
    return company_data

# matching, filtering out people with missing job entries as of 2016-1-1
# begin matching on job titles, prepare data by dropping irrelevant names

drop = ['length', 'gender', 'primary',
        'primary_weight', 'secondary', 'secondary_weight',
        'city', 'country', 'education', 'elite', '.', '??',
        '/', 'department', 'exchange',
        'public', 'location_company', 'educational', 'degree', 'elite_education',
        'major', 'department', 'FIGI', 'last_update', 'industry', 'birth', 'company']
matching_data = {company_name: company_data.drop(labels=drop, axis=1) for company_name, company_data in data.items()}


def job_2008(company_name, company_data):
    """"
    Return each user's job at the given company as of 2008-01-01
    """
    date_2008 = pd.to_datetime('2008-01-01')
    missing_start = pd.to_datetime('1900-01-01')
    missing_end = pd.to_datetime('2018-01-01')

    company_tickers = {'db': 'DB', 'leh': 'LEH', 'gs': 'GS', 'ms': 'MS^E', 'ubs': 'UBS'}

    x = company_data
    mask = ~((x['start'] == missing_start) & (x['end'] == missing_end)) & \
           (x['start'] < date_2008) & \
           (x['end'] > date_2008) & \
           (x['ticker'] == company_tickers[company_name])
    return company_data[mask]


job_as_of_2008 = {company_name: job_2008(company_name, company_data) for company_name, company_data in
                  matching_data.items()}

all_data = pd.concat(job_as_of_2008.values())
# only person missing a role in the entire data set
all_data = all_data.drop(11512)

# begin extracting job titles
directors = set(all_data[(all_data.role.str.contains(r'director|MD,md', case=False))
                         | (all_data.role.str.match(r'ed|md', case=False))].user)
all_roles = directors.copy()

analysts = set(all_data[all_data.role.str.contains('analyst|Anaylst', case=False)].user).difference(all_roles)
all_roles = all_roles.union(analysts)

vps = set(all_data[all_data.role.str.contains('president|vp', case=False)].user).difference(all_roles)
all_roles = all_roles.union(vps)

assocs = set(all_data[all_data.role.str.contains('associate', case=False)].user).difference(all_roles)
all_roles = all_roles.union(assocs)

accountants = set(
    all_data[all_data.role.str.contains('accountant|account executive|accounting', case=False)].user).difference(
    all_roles)
all_roles = all_roles.union(accountants)

consultants = set(all_data[all_data.role.str.contains('consultant', case=False)].user).difference(all_roles)
all_roles = all_roles.union(consultants)

missing = set(all_data[all_data.role.str.match(r'-|\?|\.', case=False)].user).difference(all_roles)
all_roles = all_roles.union(missing)

developers = set(
    all_data[all_data.role.str.contains(r'developer|engineer|system administrator', case=False)].user).difference(
    all_roles)
all_roles = all_roles.union(developers)

interns = set(all_data[all_data.role.str.contains('intern|trainee|apprentice', case=False)].user).difference(all_roles)
all_roles = all_roles.union(interns)

specialists = set(
    all_data[all_data.role.str.contains('specialist|administrator|research|expert', case=False)].user).difference(
    all_roles)
all_roles = all_roles.union(specialists)

sales = set(all_data[all_data.role.str.contains('sales', case=False)].user).difference(all_roles)
all_roles = all_roles.union(sales)

traders = set(all_data[all_data.role.str.contains(r'trader|trading|Portfolio Management', case=False)].user).difference(
    all_roles)
all_roles = all_roles.union(traders)

bankers = set(all_data[all_data.role.str.contains(r'banking|banker|finance', case=False)].user).difference(all_roles)
all_roles = all_roles.union(bankers)

controllers = set(all_data[all_data.role.str.contains('controller', case=False)].user).difference(all_roles)
all_roles = all_roles.union(controllers)

partners = set(all_data[all_data.role.str.contains('partner', case=False)].user).difference(all_roles)
all_roles = all_roles.union(partners)

counsels = set(all_data[all_data.role.str.contains('counsel', case=False)].user).difference(all_roles)
all_roles = all_roles.union(counsels)

recruiters = set(all_data[all_data.role.str.contains('recruiter|human resources', case=False)].user).difference(
    all_roles)
all_roles = all_roles.union(recruiters)

advisors = set(all_data[all_data.role.str.contains('advisor|adviseur', case=False)].user).difference(all_roles)
all_roles = all_roles.union(advisors)

assistants = set(
    all_data[all_data.role.str.contains('assistant|support|services|receptionist', case=False)].user).difference(
    all_roles)
all_roles = all_roles.union(assistants)

managers = set(all_data[all_data.role.str.contains(
    r'manager|supervisor|team lead|head|lead|coordinator|representative|process executive',
    case=False)].user).difference(all_roles)
all_roles = all_roles.union(managers)

others = set(all_data.user).difference(all_roles)

# zip all sets and all job title names
all_sets = [directors, analysts, vps, assocs, advisors, assistants, consultants, managers, missing, developers, interns,
            specialists, sales, traders, bankers, controllers, partners, counsels, recruiters, accountants, others]
job_titles = ['director', 'analyst', 'vp', 'assoc', 'advisor', 'assistant', 'consultant', 'manager', 'missing',
              'developer', 'intern', 'specialist', 'sale', 'trader', 'banker', 'controller', 'parnter', 'counsel',
              'recruiter', 'accountant', 'other']

zipped = list(zip(all_sets, job_titles))


def to_dict(dictionary, users, job_title):
    """Map users to job_title in the given dictionary"""
    for user in users:
        dictionary.update({user: job_title})


full_mapping = {}
[to_dict(full_mapping, x, y) for x, y in zipped]
full_mapping.update({'c0a3eb6a-59db-3a30-8a39-99a7cc8b9ce1': 'specialist'})
full_mapping.update({'5f425323-1cdf-3e81-a08e-35b483c42da9': 'missing'})

import statsmodels.discrete.discrete_model as sm

# prepare data for regression by dropping irrelevant names
drop = ['length', 'name', 'industry',
        'primary_weight', 'secondary', 'secondary_weight', 'elite_education',
        'city', 'country', '.', '??',
        '/', 'department', 'exchange',
        'public', 'location_company',
        'major', 'department', 'FIGI', 'last_update', 'company', 'normalized_company', 'educational', 'degree']

regression_data = {company_name: company_data.drop(labels=drop, axis=1) for company_name, company_data in data.items()}

# additional step of filtering out those who don't have a job entry on 2016-1-1
all_data_2016 = pd.concat(regression_data.values())
mask = (all_data_2016['start'] <= pd.to_datetime('2016-1-1')) & (
            all_data_2016['end'] >= pd.to_datetime('2016-1-1')) & ~(
            (all_data_2016['start'] == pd.to_datetime('1900-01-01')) & (
                all_data_2016['end'] == pd.to_datetime('2018-01-01'))) & (
           ~all_data_2016['ticker'].isin(['UNIVERSITY', 'SCHOOL']))
employed_2016 = all_data_2016[mask]
employ_2016_users = list(employed_2016.user.unique())

regression_data = {company_name: job_2008(company_name, company_data) for company_name, company_data in
                   regression_data.items()}

non_lehman = pd.concat([regression_data['db'], regression_data['gs'], regression_data['ms'], regression_data['ubs']])
non_lehman['is_lehman'] = 0

lehman = regression_data['leh'].copy()
lehman['is_lehman'] = 1

all_data = pd.concat([lehman, non_lehman])

all_data = all_data[all_data.user.isin(employ_2016_users)]

# fill in missing births to the median date, 1976
index = all_data[all_data.birth.isin(['None', '2000'])].index
all_data.loc[index, ['birth']] = '1976'

# data deemed informative by information gain. Missing is coded as '-1'
informative_skills = ['Operations Management', 'Insurance', 'Business Development', 'Product Management', '-1']
# convert uninformative skills to '0'
not_informative = ~all_data.primary.isin(informative_skills)
all_data.loc[not_informative, 'primary'] = 0

# make sure typing is consistent for each category
X = all_data[['birth', 'gender', 'primary', 'education', 'elite']].copy()
X['education'] = X['education'].apply(str)
X['gender'] = X['gender'].apply(str)
X['birth'] = X['birth'].astype(int)
X['elite'] = X['elite'].astype(int)
X = pd.get_dummies(data=X, drop_first=True)
X = sm.tools.add_constant(X)

y = all_data['is_lehman']

# regress y on X
logit = sm.Logit(y, X)
results = logit.fit()

# get propensities
all_data['propensity'] = results.predict(X)

# applying job user to job category mapping from earlier
all_data['job_category'] = all_data.user.apply(lambda x: full_mapping[x])

# Begin matching process. Map each user to its propensity
user_to_propensity = dict(zip(all_data.user, all_data.propensity))

# get lehman and non-lehman guys
lehman = all_data[all_data['is_lehman'] == 1]
non_lehman = all_data[all_data['is_lehman'] == 0]


def get_closest(row):
    # return user ID of closest match with the same job title
    role = row.job_category
    score = row.propensity
    others_by_role = non_lehman[non_lehman.job_category == role].set_index('user')
    return np.absolute(others_by_role['propensity'] - score).idxmin()


# get closest match for each lehman guy
lehman['match'] = lehman.apply(get_closest, axis=1)
lehman['match_propensity'] = lehman.match.apply(lambda x: user_to_propensity[x])
lehman_to_match = dict(zip(lehman.user, lehman.match))


# naive proportion in finance
def mask(company_data):
    """
    Return values in the time range with start before '2016-1-1' and end after '2016-1-1'.

    Excludes values that don't have a start or end time.
    """
    mask = (company_data['start'] <= pd.to_datetime('2016-1-1')) & (
                company_data['end'] >= pd.to_datetime('2016-1-1')) & ~(
                (company_data['start'] == pd.to_datetime('1900-01-01')) & (
                    company_data['end'] == pd.to_datetime('2018-01-01'))) & (
               ~company_data['ticker'].isin(['UNIVERSITY', 'SCHOOL']))
    return company_data[mask]


def filter_and_mask(company_data):
    # combines filter and mask
    filtered = filter_manual(company_data)
    return mask(filtered)


drop = ['length', 'gender', 'primary',
        'primary_weight', 'secondary', 'secondary_weight',
        'city', 'country', 'education', 'elite', '.', '??',
        '/', 'department', 'exchange',
        'public', 'location_company', 'educational', 'degree', 'elite_education',
        'major', 'department', 'FIGI', 'last_update']

finance_data = {company_name: company_data.drop(labels=drop, axis=1) for company_name, company_data in data.items()}


def prop_finance(company_data):
    # exclude values with tickers in the categories
    copy = company_data.copy()
    users = company_data.groupby('user')
    total_users = len(users)
    # get each person's most recent job
    recent_jobs = users.first()
    # sum the people who stayed in finance industries
    stayed_finance = sum(recent_jobs['industry'].str.startswith('52', na=False))
    return stayed_finance, total_users, stayed_finance / total_users


# Apply mappings for missing industries with the dictionary, then mask to look only at job entries that start before
# '2016-1-1' and end after '2016-1-1'.
filtered_data = {company_name: filter_and_mask(company_data) for company_name, company_data in finance_data.items()}

# calculate the proportion that stayed in finance as of 2016-01-01
prop_stayed_finance = {company_name: prop_finance(company_data) for company_name, company_data in filtered_data.items()}
prop_stayed_finance

# # same as before, but don't filter the data with the dictionary
# filtered_data = {company_name: mask(company_data) for company_name, company_data in finance_data.items()}
# prop_unfiltered = {company_name: prop_finance(company_data) for company_name, company_data in filtered_data.items()}
# prop_unfiltered

# Matched proportion finance
lehman = filtered_data['leh'].copy()
lehman['match'] = lehman.user.apply(lambda x: lehman_to_match[x])

matched_users = list(lehman.match.unique())

non_lehman = pd.concat([filtered_data['db'], filtered_data['gs'], filtered_data['ms'], filtered_data['ubs']])
matches = non_lehman[non_lehman.user.isin(matched_users)]
matches = mask(matches)
matches['stayed_finance'] = matches['industry'].str.startswith('52', na=False)
user_to_stayed_finance = dict(zip(matches.user, matches.stayed_finance))
lehman_most_recent = lehman.groupby('user').first()
lehman_most_recent['match_stayed_finance'] = lehman_most_recent.match.apply(lambda x: user_to_stayed_finance[x])
lehman_stayed = sum(lehman_most_recent.industry.str.startswith('52', na=False))
lehman_total = len(lehman_most_recent)
lehman_prop = lehman_stayed / lehman_total
match_stayed = sum(lehman_most_recent.match_stayed_finance)
match_total = len(lehman_most_recent)
match_prop = match_stayed / match_total
{'lehman': (lehman_stayed, lehman_total, lehman_prop), 'match': (match_stayed, match_total, match_prop)}
