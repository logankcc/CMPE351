
from getData import get_data, preprocess_data

if __name__ == "__main__":
    state = "Montana"
    df_Data = get_data(state=state)

    train, val, test = preprocess_data(df_Data=df_Data)

    # train/test model (comment out what is not needed)
    # train_model(n_epochs=1, train_set=train, val_set=val)
    # test_model(test, "d_weights_epoch_10.pt")

    print("\ntest")
    print(train.head())
    print(train.columns)
    print(train.shape)
    print("\nval")
    print(val.head())
    print(val.columns)
    print(val.shape)
    print("\ntest")
    print(test.head())
    print(test.columns)
    print(test.shape)

