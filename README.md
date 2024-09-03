Calling Syntax, the first file is the name of the holter file (200 hz) which includes the .mrk and .ecg and the second is the name of the everbeat file (in this case 19.eb)

python3 align_gsm03.py 0000019 19

The first plot should look like "Alignment-image.png" the second plot should look like "Alignement-scoring-image.png"

The larger the delta between the highest peak in the scoring image and the second highest peak, the more likely this alignment is correct and optimal.


Align csv files:
For testing align quality we need 2 csv files, downloaded from everbeat dashboard. Script call example

`python3 align_csv.py db ref`
