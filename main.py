from data_collector import DataCollector

if __name__ == "__main__":
    label = input("Ingresa la etiqueta del gesto a recolectar: ")
    collector = DataCollector()
    collector.collect(label)
