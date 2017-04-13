#!/bin/bash

mkdir -p resources/datasets
cd resources/datasets

# Download embedding matrices
wget https://github.com/kandidat-highlights/data/raw/master/Glove/vectors100d.tar.gz
tar -xzf vectors100d.tar.gz
rm vectors100d.tar.gz 

wget https://github.com/kandidat-highlights/data/raw/master/Glove/vectors150d.tar.gz
tar -xzf vectors150d.tar.gz
rm vectors150d.tar.gz

# Download datasets
wget https://github.com/kandidat-highlights/data/raw/master/allVotes/data_top50_users_subreddit_title_all_votes.tar.gz
tar -xzf data_top50_users_subreddit_title_all_votes.tar.gz
rm data_top50_users_subreddit_title_all_votes.tar.gz

wget https://github.com/kandidat-highlights/data/raw/master/allVotes/data_top5_users_subreddit_title_all_votes.tar.gz
tar -xzf data_top5_users_subreddit_title_all_votes.tar.gz
rm data_top5_users_subreddit_title_all_votes.tar.gz

wget https://github.com/kandidat-highlights/data/raw/master/top50/data_top50_users_subreddit_title.tar.gz
tar -xzf data_top50_users_subreddit_title.tar.gz
rm data_top50_users_subreddit_title.tar.gz

cd ../../
