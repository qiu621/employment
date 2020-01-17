import pandas as pd
import numpy as np
import statsmodels.discrete.discrete_model as sm


def main(csv_file):
    """
    Given an appropriate csv_file, output the relevant columns.

    Returns df with all columns
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
    employ_data.columns = name
    return employ_data


# data without datetime features, and none values for some dates
raw_data = {'db': main('./Data/DB_profiles.csv'),
            'gs': main('./Data/GS_profiles.csv'),
            'leh': main('./Data/LEH_profiles.csv'),
            'ms': main('./Data/MS_profiles.csv'),
            'ubs': main('./Data/UBS_profiles.csv')
            }


def standardize_dates(company):
    """
    Converts start date and end date to datetime objects, and converts None values to the minimum or maximum date.

    Returns the modified dataframe
    """
    company_data = raw_data[company].copy()
    company_data['start'] = company_data['start'].str.replace('None', '1900-01-01')
    company_data['end'] = company_data['end'].str.replace('None', '2018-01-01')
    company_data['start'] = pd.to_datetime(company_data['start'])
    company_data['end'] = pd.to_datetime(company_data['end'])
    return company_data


# set up dictionary to hold data for each company
for company in raw_data.keys():
    raw_data[company] = standardize_dates(company)


def get_users(company_name, company_data):
    """"
    Returns the users who worked at the given company as 2008-01-01 (so as to be robust to future extractions of data)
    """
    date_2008 = pd.to_datetime('2008-01-01')
    missing_start = pd.to_datetime('1900-01-01')
    missing_end = pd.to_datetime('2018-01-01')

    x = company_data

    company_tickers = {'db': 'DB', 'leh': 'LEH', 'gs': 'GS', 'ms': 'MS^E', 'ubs': 'UBS'}
    # conditions: start and end not both missing, worked before/after 2008-01-01, ticker matches company
    mask = ~((x['start'] == missing_start) & (x['end'] == missing_end)) & \
           (x['start'] < date_2008) & \
           (x['end'] > date_2008) & \
           (x['ticker'] == company_tickers[company_name])
    return company_data[mask]['user'].unique()


# gets the user_ids within each company that match the conditioning, before and and after
#   2008-01-01, exclusive
users = {company_name: get_users(company_name, company_data) for company_name, company_data in raw_data.items()}

# Make sure that our dataset only contains people that worked at a given company as of 2008-01-01
data = {}
for company, company_data in raw_data.items():
    company_users = users[company]
    data[company] = company_data[company_data['user'].isin(company_users)]

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

# prepare data for regression by dropping irrelevant names
drop = ['length', 'name', 'industry',
        'primary_weight', 'secondary', 'secondary_weight', 'elite_education',
        'city', 'country', '.', '??',
        '/', 'department', 'exchange',
        'public', 'location_company',
        'major', 'department', 'FIGI', 'last_update', 'company', 'normalized_company', 'educational', 'degree']

regression_data = {company_name: company_data.drop(labels=drop, axis=1) for company_name, company_data in data.items()}

regression_data = {company_name: job_2008(company_name, company_data) for company_name, company_data in
                   regression_data.items()}

non_lehman = pd.concat([regression_data['db'], regression_data['gs'], regression_data['ms'], regression_data['ubs']])
non_lehman['is_lehman'] = 0

lehman = regression_data['leh'].copy()
lehman['is_lehman'] = 1

all_data = pd.concat([lehman, non_lehman])

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
