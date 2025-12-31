from pathlib import Path
import pandas as pd
import numpy as np
import math
import re
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TEMP_DIR = BASE_DIR / "data" / "temp"
FIGURES_DIR = BASE_DIR / "figures"


AVAILABILITY_PATH = RAW_DATA_DIR / "availability_rate.csv"
REVIEW_COUNTS_PATH = RAW_DATA_DIR / "review_counts.csv"


FILLED_DATA_PATH = TEMP_DIR / "filled_airbnb_data.csv"
FILLED_TEST_PATH = TEMP_DIR / "filled_test.csv"
CLEANED_V1_PATH = TEMP_DIR / "airbnb_cleaned_categorized_v1.csv"
CLEANED_TEST_PATH = TEMP_DIR / "airbnb_cleaned_categorized_test.csv"
FINAL_TRAIN_PATH = PROCESSED_DIR / "airbnb_train_last.csv"
FINAL_TEST_PATH = PROCESSED_DIR / "airbnb_test_last.csv"


def percent_to_float(value):
    if pd.isna(value):
        return None
    value = str(value).strip().replace('%', '')
    try:
        return float(value)
    except ValueError:
        return None

def clean_price(value):
    if pd.isna(value):
        return None
    value = str(value)
    # remove TL, $, TL symbols
    value = re.sub(r'[\$₺TL]', '', value)
    # remove thousands separator
    value = value.replace(',', '')
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return None

def extract_bath_num(x):
    if pd.isna(x):
        return np.nan
    x = x.lower()
    if 'half' in x:
        return 0.5
    try:
        return float(x.split()[0])
    except (ValueError, IndexError):
        return np.nan

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def check_column_types(df, label=""):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    category_cols = df.select_dtypes(include=['category']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols + object_cols + category_cols + bool_cols]
    
    print(f"\n--- Column Types {label} ---")
    print(f"Numeric  ({len(numeric_cols)}): {numeric_cols}")
    print(f"Object   ({len(object_cols)}): {object_cols}")
    print(f"Category ({len(category_cols)}): {category_cols}")
    print(f"Boolean  ({len(bool_cols)}): {bool_cols}")
    print(f"Other    ({len(other_cols)}): {other_cols}")

def plot_correlation_matrix(df, title="Correlation Matrix", filename="correlation_matrix.png"):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return
    plt.figure(figsize=(20, 16))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_distributions(df, cols, n_cols=5, filename="distributions.png"):
    if not cols:
        return
    n_rows = math.ceil(len(cols) / n_cols)
    plt.figure(figsize=(16, 4 * n_rows))
    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_boxplots(df, cols, n_cols=3, filename="boxplots.png"):
    if not cols:
        return
    n_rows = math.ceil(len(cols) / n_cols)
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def dropping_unnecessary_columns(df):
    drop_cols = [
        'Unnamed: 0.1', 'Unnamed: 0','host_id','listing_url', 'scrape_id', 'last_scraped', 'source',
        'name', 'description', 'neighborhood_overview', 
        'picture_url', 'host_url', 'host_name', 'host_about',
        'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
        'license', 'host_verifications', 'host_location'
    ]
    
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    
    print(f"Shape before dropping: {df.shape}")
    df = df.drop(columns=cols_to_drop)
    print(f"Shape after dropping: {df.shape}")
    return df

def basic_cleaning_and_types(df):
    # convert percentage strings
    if 'host_response_rate' in df.columns:
        df['host_response_rate'] = df['host_response_rate'].apply(percent_to_float)
    if 'host_acceptance_rate' in df.columns:
        df['host_acceptance_rate'] = df['host_acceptance_rate'].apply(percent_to_float)
    
    # clean price
    if 'price' in df.columns:
        df['price'] = df['price'].apply(clean_price)
    
    # convert specific columns to boolean
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('bool')
            
    # convert host_since to datetime
    if 'host_since' in df.columns:
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        
    return df

def impute_amenity_features(df):
    # handle bathrooms
    if 'bathrooms_text' in df.columns:
        df['bathrooms'] = df['bathrooms_text'].apply(extract_bath_num)
        df['bathrooms'] = df['bathrooms'].fillna(1)
    
    # impute beds and bedrooms based on accommodates
    # if accommodates is 1, default beds and bedrooms to 1 if missing
    mask_acc1 = (df['accommodates'] == 1)
    df.loc[mask_acc1 & df['beds'].isna(), 'beds'] = 1
    df.loc[mask_acc1 & df['bedrooms'].isna(), 'bedrooms'] = 1
    
    # fill missing beds based on bedroom count (using observed ratio ~1.37)
    df['beds'] = df['beds'].fillna((df['bedrooms'] * 1.37).round())
    
    # fill missing bedrooms based on bed count
    mask_bed_missing = df['bedrooms'].isna() & df['beds'].notna()
    df.loc[mask_bed_missing, 'bedrooms'] = df.loc[mask_bed_missing, 'beds'].apply(lambda x: math.ceil(x / 1.37))
    
    # fill remaining with mode
    mode_bedrooms = df['bedrooms'].mode()[0] if not df['bedrooms'].mode().empty else 1
    df['bedrooms'] = df['bedrooms'].fillna(mode_bedrooms)
    
    mode_beds = df['beds'].mode()[0] if not df['beds'].mode().empty else 1
    df['beds'] = df['beds'].fillna(mode_beds)
    
    # ensure minimum 1 for beds/bedrooms
    df['beds'] = df['beds'].replace(0, 1)
    df['bedrooms'] = df['bedrooms'].replace(0, 1)
    
    return df

def impute_host_info(df):
       
    df['host_listings_count'] = df['host_listings_count'].fillna(df['calculated_host_listings_count'])
    df['host_total_listings_count'] = df['host_total_listings_count'].fillna(df['host_listings_count'])
    
    # default host_since for missing values
    df['host_since'] = df['host_since'].fillna(pd.Timestamp('2025-01-01'))
    
    # handle response and acceptance rates
    # if all three major host metrics are missing, mark as unknown/0
    missing_all_three = df[['host_response_time', 'host_response_rate', 'host_acceptance_rate']].isna().all(axis=1)
    df.loc[missing_all_three, "host_response_time"] = "unknown"
    df.loc[missing_all_three, 'host_response_rate'] = 0
    df.loc[missing_all_three, 'host_acceptance_rate'] = 0
    
    # fill remaining missing with median/mode
    df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].median())
    df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna(df['host_acceptance_rate'].median())
    
    if not df['host_response_time'].mode().empty:
        df['host_response_time'] = df['host_response_time'].fillna(df['host_response_time'].mode()[0])
    
    return df

def impute_reviews_with_mean(train_df, test_df):
    review_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value'
    ]
    
    for col in review_cols:
        mean_val = train_df[col].mean()
        train_df[col] = train_df[col].fillna(mean_val)
        test_df[col] = test_df[col].fillna(mean_val)
            
    print("Review score imputation (Mean) completed.")
    return train_df, test_df

def feature_engineering_basic(df):
    
    # Average review score across all categories
    review_cols = [
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]
    df['average_review_score'] = df[review_cols].mean(axis=1)
    
    # host experience in years
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['host_experience_years'] = 2025 - df['host_since'].dt.year
    
    # amenities count
    df['amenities_count'] = df['amenities'].apply(lambda x: len(ast.literal_eval(x)) if pd.notnull(x) and isinstance(x, str) else 0)
    
    # minimum nights categorization
    df['is_long_term'] = (df['minimum_nights'] >= 30).astype(int)
    
    return df

def feature_engineering_amenities(df):
    # prepare amenities string for searching
    if 'amenities' in df.columns:
        # use a list for Counter analysis
        df['amenities_list'] = df['amenities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        
        # counter analysis
        all_items = [item for sublist in df['amenities_list'] for item in sublist]
        counts = Counter(all_items)
        print("\n--- Most seen features (top 20) ---")
        for item, count in counts.most_common(20):
            print(f"{item}: {count}")
            
        amenities_str = df['amenities_list'].astype(str).str.lower()
        
        # Gym / Fitness
        df['has_gym'] = amenities_str.str.contains('gym|fitness|weight').astype(int)
        
        # Pool (excluding billiard/pool tables)
        df['has_pool'] = (amenities_str.str.contains('pool|swim|beach') & 
                          ~amenities_str.str.contains('table')).astype(int)
        
        # Spa / Relaxing
        df['has_spa'] = amenities_str.str.contains('sauna|jacuzzi|hot tub|whirlpool|bathtub').astype(int)
        
        # Air Conditioning
        df['has_ac'] = amenities_str.str.contains('air cond|ac unit').astype(int)
        
        # View
        df['has_view'] = amenities_str.str.contains('view').astype(int)
        
        # drop temporary list column
        df = df.drop(columns=['amenities_list'])
        
    return df

def feature_engineering_location(df):
    
    pois = {
        'bosphorus': (41.0422, 29.0083),
        'sultanahmet': (41.0054, 28.9768),
        'taksim': (41.0370, 28.9850),
        'kadikoy': (40.9927, 29.0230),
        'besiktas': (41.0430, 29.0050),
        'galata': (41.0256, 28.9744),
        'eminonu': (41.0175, 28.9711),
        'airport_ist': (41.2608, 28.7418),
        'airport_saw': (40.8986, 29.3092),
    }
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        lat, lon = df['latitude'], df['longitude']
        
        # calculate distance for each POI
        for name, coords in pois.items():
            df[f'dist_{name}'] = haversine(lat, lon, coords[0], coords[1])
            
        # summary distance features
        tourist_cols = ['dist_sultanahmet', 'dist_taksim', 'dist_galata', 'dist_eminonu', 'dist_besiktas']
        df['min_dist_tourist'] = df[tourist_cols].min(axis=1)
        df['avg_dist_tourist'] = df[tourist_cols].mean(axis=1)
        df['min_dist_airport'] = df[['dist_airport_ist', 'dist_airport_saw']].min(axis=1)
        
    return df

def handle_outliers_and_scaling(df):
    
    for col in ['maximum_nights', 'minimum_nights']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
            
        # clip review scores to max 5
    review_cols = [
        'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location', 'review_scores_value',
        'review_scores_rating'
    ]
    for col in review_cols:
        if col in df.columns:
            df[col] = df[col].clip(upper=5)
            
    # log transformations for host and physical counts
    log_cols = [
        'host_listings_count', 'host_total_listings_count', 'calculated_host_listings_count',
        'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms', 'bathrooms', 'bedrooms', 'beds', 'accommodates'
    ]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            
    return df

def categorical_encoding(train_df, test_df):
    top_10_types = train_df['property_type'].value_counts().nlargest(10).index
    train_df['property_type_clean'] = train_df['property_type'].apply(lambda x: x if x in top_10_types else 'Other')
    test_df['property_type_clean'] = test_df['property_type'].apply(lambda x: x if x in top_10_types else 'Other')
    
    # neighbourhood rank encoding (fixed: use count to avoid leakage)
    # rank by number of listings (popularity) instead of price to avoid target leakage
    listing_counts = train_df['neighbourhood_cleansed'].value_counts()
    rank_mapping = listing_counts.rank(method='dense', ascending=True).astype(int).to_dict()
    
    train_df["neighbourhood_popularity_rank"] = train_df['neighbourhood_cleansed'].map(rank_mapping).fillna(0)
    test_df["neighbourhood_popularity_rank"] = test_df['neighbourhood_cleansed'].map(rank_mapping).fillna(0)
    
    # host response time ordinal encoding
    response_time_order = {
        'within an hour': 4, 
        'within a few hours': 3,
        'within a day': 2,
        'a few days or more': 1,
        'unknown': 0  
    }
    train_df['host_response_time_encoded'] = train_df['host_response_time'].map(response_time_order).fillna(0)
    test_df['host_response_time_encoded'] = test_df['host_response_time'].map(response_time_order).fillna(0)
    
    # one-hot encoding
    train_df = pd.get_dummies(train_df, columns=['room_type', 'property_type_clean'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['room_type', 'property_type_clean'], drop_first=True)
    
    return train_df, test_df

def main():
    print("Starting Preprocessing Pipeline...")
    
    # load data
    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "test.csv")


    try:
        calendar = pd.read_csv(AVAILABILITY_PATH)
        review_counts = pd.read_csv(REVIEW_COUNTS_PATH)
        print("External data files loaded successfully.")
        
        #  (Availability)
        calendar_cols = ['listing_id', 'weekend_availability_rate', 'weekday_availability_rate']
        train = pd.merge(train, calendar[calendar_cols], left_on='id', right_on='listing_id', how='left')
        test = pd.merge(test, calendar[calendar_cols], left_on='id', right_on='listing_id', how='left')
        
        # (Review Bins)
        review_cols = ['listing_id', 'num_reviews_bin']
        train = pd.merge(train, review_counts[review_cols], left_on='id', right_on='listing_id', how='left')
        test = pd.merge(test, review_counts[review_cols], left_on='id', right_on='listing_id', how='left')
        
        
        cols_to_drop = [c for c in train.columns if 'listing_id' in c]
        train.drop(columns=cols_to_drop, inplace=True)
        test.drop(columns=cols_to_drop, inplace=True)
        
        
        train['num_reviews_bin'] = train['num_reviews_bin'].fillna(0)
        test['num_reviews_bin'] = test['num_reviews_bin'].fillna(0)
        
        
        train['weekend_availability_rate'] = train['weekend_availability_rate'].fillna(train['weekend_availability_rate'].median())
        train['weekday_availability_rate'] = train['weekday_availability_rate'].fillna(train['weekday_availability_rate'].median())
        test['weekend_availability_rate'] = test['weekend_availability_rate'].fillna(train['weekend_availability_rate'].median())
        test['weekday_availability_rate'] = test['weekday_availability_rate'].fillna(train['weekday_availability_rate'].median())

    except FileNotFoundError as e:
        print(f"Warning: External files not found. Error: {e}")    
    print(f"Initial Train Shape: {train.shape}, Test Shape: {test.shape}")
    
    # initial drop and basic cleaning
    train = dropping_unnecessary_columns(train)
    test = dropping_unnecessary_columns(test)
    
    train = basic_cleaning_and_types(train)
    test = basic_cleaning_and_types(test)
    
    train = train.drop_duplicates()
    
    # initial visualization
    plot_correlation_matrix(train, "Initial Numeric Correlation Heatmap", "initial_correlation.png")
    
    # impute physical features
    train = impute_amenity_features(train)
    test = impute_amenity_features(test)
    
    # impute host info
    train = impute_host_info(train)
    test = impute_host_info(test)
    
    # impute review scores (using xgboost)
    train, test = impute_reviews_with_mean(train, test)
    
    # drop price nas and irrelevant columns
    train = train.dropna(subset=["price"])
    
   
    # simple outlier cleaning 
    before_outliers = len(train)
    train = train[train['price'] > 0]
    price_cap = train['price'].quantile(0.995)
    train = train[train['price'] <= price_cap]
    
    # clean extreme bedroom/bed counts as they are often typos
    bed_cap = train['bathrooms'].quantile(0.995)
    train = train[train['bathrooms'] <= bed_cap]
    
  
    
    print(f"Outlier Cleaning: Kept 99.9% of data. Removed {before_outliers - len(train)} rows total")

    final_drop = ['neighbourhood_group_cleansed', 'neighbourhood', 'bathrooms_text']
    train = train.drop(columns=[c for c in final_drop if c in train.columns])
    test = test.drop(columns=[c for c in final_drop if c in test.columns])
    
    # intermediate save
    train.to_csv(FILLED_DATA_PATH, index=False)
    test.to_csv(FILLED_TEST_PATH, index=False)
    
    # basic feature engineering
    train = feature_engineering_basic(train)
    test = feature_engineering_basic(test)
    
    # visualization of processed numerical columns
    numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    plot_distributions(train, numeric_cols, filename="intermediate_distributions.png")
    plot_boxplots(train, numeric_cols, filename="intermediate_boxplots.png")
    
    # outlier handling for hotels
    hotel_types = [
        'Room in aparthotel', 'Room in bed and breakfast', 'Room in boutique hotel',
        'Room in heritage hotel', 'Room in hostel', 'Room in hotel', 
        'Shared room in hostel', 'Shared room in hotel', 'Private room in hostel'
    ]
    
    def cap_hotel_features(row):
        if row['property_type'] in hotel_types:
            row['accommodates'] = min(row['accommodates'], 5)
            row['beds'] = min(row['beds'], 5)
            row['bedrooms'] = min(row['bedrooms'], 3)
            row['bathrooms'] = min(row['bathrooms'], 3)
        return row
    
    train = train.apply(cap_hotel_features, axis=1)
    test = test.apply(cap_hotel_features, axis=1)
    
    # outlier handling & log transformation
    train = handle_outliers_and_scaling(train)
    test = handle_outliers_and_scaling(test)
    
    # amenity-specific features
    train = feature_engineering_amenities(train)
    test = feature_engineering_amenities(test)
    
    # location-specific features (POI distances)
    print("Calculating POI distances...")
    train = feature_engineering_location(train)
    test = feature_engineering_location(test)
    
    # categorical encoding
    train, test = categorical_encoding(train, test)
    
    # final column selection and cleanup
    review_cols_to_drop = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value'
    ]

    dist_detail_cols = [
        'dist_taksim', 'dist_galata', 'dist_sultanahmet', 
        'dist_besiktas', 'dist_eminonu', 'dist_kadikoy', 'dist_bosphorus'
    ]
    airport_cols = ['dist_airport_ist', 'dist_airport_saw']

    host_count_cols = [
        'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms',
        'host_listings_count',
        'host_total_listings_count'
    ]
    
    cols_to_drop = review_cols_to_drop + host_count_cols + dist_detail_cols + airport_cols +  [
        'host_since', 'maximum_nights', 'amenities', 'property_type', 'neighbourhood_cleansed',
        'latitude', 'longitude', 'host_response_time', 'host_response_rate', 'minimum_nights','weekday_availability_rate','min_dist_tourist'
    ]
    
    train = train.drop(columns=[c for c in cols_to_drop if c in train.columns])
    test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])
    
            # target variable preparation
    # log(1+p) transformation for RMSLE optimization
    train['log_price'] = np.log1p(train['price'])
    
    # target capping at 99th percentile for training stability
    cap_value = train['price'].quantile(0.99)
    train['log_price_cap'] = np.log1p(train['price'].clip(upper=cap_value))
    
    # align columns
    # keep 'price', 'log_price', and 'log_price_cap' in train for the model to choose from
    target_cols = ['price', 'log_price', 'log_price_cap']
    
    # ensure only numeric/bool columns remain (excluding targets for a moment)
    features_train = train.drop(columns=target_cols, errors='ignore').select_dtypes(include=[np.number, 'bool'])
    features_test = test.select_dtypes(include=[np.number, 'bool'])
    
    # drop any remaining unnamed or index columns from features
    drop_patterns = ['Unnamed', 'index']
    cols_to_drop_final = [c for c in features_train.columns if any(pat in c for pat in drop_patterns)]
    features_train = features_train.drop(columns=cols_to_drop_final)
    features_test = features_test.drop(columns=[c for c in cols_to_drop_final if c in features_test.columns])
    
    # align test columns with train features
    train_feature_names = features_train.columns.tolist()
    features_test = features_test.reindex(columns=train_feature_names, fill_value=0)
    
    # reassemble final dataframes
    train = pd.concat([features_train, train[target_cols]], axis=1)
    test = features_test
    
    # final check: no nas/infs (ensuring parity with original script's cleaning)
    train = train.replace([np.inf, -np.inf], np.nan).fillna(0)
    test = test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # final visualizations
    plot_correlation_matrix(train, "Final Correlation Matrix", "final_correlation.png")
    final_numeric = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
    plot_distributions(train, final_numeric, filename="final_distributions.png")
    plot_boxplots(train, final_numeric, filename="final_boxplots.png")
    
        # save final data
    train.to_csv(FINAL_TRAIN_PATH, index=False, encoding='utf-8-sig')
    test.to_csv(FINAL_TEST_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\nFinal Train Shape: {train.shape}, Final Test Shape: {test.shape}")
    print(f"Processed data saved to {FINAL_TRAIN_PATH} and {FINAL_TEST_PATH}")

if __name__ == "__main__":
    main()
