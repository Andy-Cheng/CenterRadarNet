# this file define the domain specific settings for the kradar dataset

# Three domain splits defined in the kradar dataset paper: https://arxiv.org/pdf/2206.08171.pdf

weather1 = dict(
    train=['normal', 'overcast'],
    val=['normal', 'overcast'],
    test1=['rain', 'fog'],
    test2=['sleet', 'lightsnow', 'heavysnow']
)