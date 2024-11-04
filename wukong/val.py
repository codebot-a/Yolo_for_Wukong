from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n.pt")  # load an official model
    model = YOLO("./boss_plus_wukong.pt")  # load a custom model

    # Validate the model
    metrics = model.val(data="./yolo_wukong/wukong/my_train.yaml")  # no arguments needed, dataset and settings remembered
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # a list contains map50-95 of each category