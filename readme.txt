**Bracket Challenge Predictor** 
Purpose: For fun
Sample Implement sklearn-kit as core libraries. The models built in this way will be save for using in future application.

**Required (Basic Numpy, Pandas, Matplotlib and Scipy) **

    Flask
    Sklearn
    Keras + TensorFlow
    Seaborn
    patsy


**Result** 
Using regular season data and calculate different between two teams to predict playoff outcome provide roughtly 62% accuracy in playoff game. **The accuracy calculate from Bracket picked, if picked team disqualify since first round, the rest round result will be immediately treated as invalid.**

*Accuracy is not satisfy enough to use in actual prediction application.*

**Future improvement** 
Consider other data variables apart from regular season team statistics and player statistic - key player , coach experience, team synergies (played together for long time)
A lot of unpredictable factor in playoff, since it play only one game not 7 games like NBA.
A lot of expectation happenned in first round, For example Michigan which is 2nd seed lost to 15nd seed Middle Tenn.

**Running**

predictor

    python predict.py


**Prediction file**

    challenge_v1.pdf