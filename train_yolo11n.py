from ultralytics import YOLO

def trainYOLO():
    model = YOLO("yolo11n.pt")

    # dataset="polyhedra/polyhedra.yaml"
    dataset="statue/statue.yaml"

    # result="runs/polyhedra"
    result="runs/statue"

    model.train(
        data=dataset,
        epochs=100,
        imgsz=640,
        batch=-1,
        patience=30,
        project=result,
        name="yolo11n_v1"
    )

if __name__ == "__main__":
    trainYOLO()
