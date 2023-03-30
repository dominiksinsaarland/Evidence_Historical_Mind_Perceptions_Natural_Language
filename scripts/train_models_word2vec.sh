for year in 1820 1830 1840 1850 1860 1870 1880 1890 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010
do
  echo $year
  cat coha/$year/* > coha/$year/all_$year.txt
  python coha-code/train_model_word2vec.py $year
done




