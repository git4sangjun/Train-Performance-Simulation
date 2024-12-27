import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TrainData:
    def __init__(self, mass, max_power, braking_force):
        self.mass = mass
        self.max_power = max_power
        self.braking_force = braking_force

class SegmentData:
    def __init__(self, length, gradient, curve_radius, station_name, stop_time):
        self.length = length
        self.gradient = gradient
        self.curve_radius = curve_radius
        self.station_name = station_name
        self.stop_time = stop_time

class TPSEngine:
    def __init__(self, train, segments, time_step=1):
        self.train = train
        self.segments = segments
        self.time_step = time_step

    def calculate_resistance(self, speed):
        a, b, c = 1000, 0.1, 0.01
        return a + b * speed + c * speed**2

    def calculate_gradient_resistance(self, gradient):
        return self.train.mass * 9.81 * (gradient / 100)

    def calculate_curve_resistance(self, curve_radius):
        if curve_radius > 0:
            return 700 / curve_radius
        return 0

    def calculate_tractive_effort(self, speed):
        return min(self.train.max_power / speed, 200000) if speed > 0 else 200000

    def calculate_braking_force(self):
        return -self.train.braking_force / self.train.mass

    def calculate_deceleration_distance(self, initial_speed, final_speed):
        deceleration = abs(self.calculate_braking_force())
        distance = (initial_speed**2 - final_speed**2) / (2 * deceleration)
        return max(0, distance)

    def run_simulation(self):
        time = 0
        speed = 0
        total_distance = 0
        results = []
        station_info = []

        for segment in self.segments:
            segment_distance = 0

            while segment_distance < segment.length:
                braking_distance = self.calculate_deceleration_distance(speed, 0)
                braking = (
                    segment.station_name and 
                    (segment.length - segment_distance <= braking_distance)
                )

                resistance = (
                    self.calculate_resistance(speed) +
                    self.calculate_gradient_resistance(segment.gradient) +
                    self.calculate_curve_resistance(segment.curve_radius)
                )
                
                tractive_effort = (
                    max(self.calculate_braking_force(), -resistance) if braking else 
                    self.calculate_tractive_effort(speed)
                )
                
                acceleration = (tractive_effort - resistance) / self.train.mass
                speed += acceleration * self.time_step
                speed = max(0, speed)
                
                distance_increment = speed * self.time_step
                segment_distance += distance_increment
                total_distance += distance_increment
                time += self.time_step
                
                results.append((time, speed * 3.6, total_distance))
                
                if braking and speed <= 0.1:
                    break

            if segment.station_name:
                station_info.append((segment.station_name, total_distance, time))
                for _ in range(segment.stop_time):
                    time += self.time_step
                    results.append((time, 0, total_distance))
                speed = 0

        return results, station_info

    def calculate_segment_stats(self, results, station_info):
        segment_stats = []
        for i in range(1, len(station_info)):
            start_station = station_info[i-1]
            end_station = station_info[i]
            start_distance = start_station[1]
            end_distance = end_station[1]
            start_time = start_station[2]
            end_time = end_station[2]

            segment_results = [r for r in results if start_distance <= r[2] <= end_distance]
            speeds = [r[1] for r in segment_results]

            distance = end_distance - start_distance
            time = end_time - start_time
            avg_speed = distance / time if time > 0 else 0
            min_speed = min(speeds) if speeds else 0
            max_speed = max(speeds) if speeds else 0
            
            energy = distance * avg_speed * 0.001

            segment_stats.append({
                'start_station': start_station[0],
                'end_station': end_station[0],
                'distance': distance,
                'avg_speed': avg_speed * 3.6,
                'min_speed': min_speed,
                'max_speed': max_speed,
                'time': time,
                'energy': energy
            })
        return segment_stats

def load_train_data(file_path):
    df = pd.read_csv(file_path)
    return TrainData(mass=df['mass'][0], max_power=df['max_power'][0], braking_force=df['braking_force'][0])

def load_segments(file_path):
    df = pd.read_csv(file_path)
    return [SegmentData(
        length=row['length'],
        gradient=row['gradient'],
        curve_radius=row['curve_radius'],
        station_name=row['station_name'] if pd.notna(row['station_name']) else None,
        stop_time=row['stop_time']
    ) for _, row in df.iterrows()]

def main():
    train_file_path = "train_data.csv"
    segments_file_path = "segments.csv"

    train_data = load_train_data(train_file_path)
    segments_data = load_segments(segments_file_path)

    engine = TPSEngine(train=train_data, segments=segments_data)
    results, station_info = engine.run_simulation()
    segment_stats = engine.calculate_segment_stats(results, station_info)

    for stat in segment_stats:
        print(f"{stat['start_station']} - {stat['end_station']}:")
        print(f"  Distance: {stat['distance']:.2f} m")
        print(f"  Average Speed: {stat['avg_speed']:.2f} km/h")
        print(f"  Minimum Speed: {stat['min_speed']:.2f} km/h")
        print(f"  Maximum Speed: {stat['max_speed']:.2f} km/h")
        print(f"  Travel Time: {stat['time']:.2f} s")
        print(f"  Energy Consumption: {stat['energy']:.2f} kWh")
        print()

    # 그래프 생성 코드 추가
    times, speeds_kph, distances_meters = zip(*results)

    plt.figure(figsize=(12, 6))
    plt.plot(distances_meters, speeds_kph, label="Speed (km/h)", color="blue")

    for station in station_info:
        plt.axvline(x=station[1], color='red', linestyle='--', label=f"{station[0]} (Stop)")
        plt.text(station[1], plt.ylim()[1], station[0], rotation=90)

    plt.xlabel("Distance (m)")
    plt.ylabel("Speed (km/h)")
    plt.title("Train Performance Simulation with Curve Resistance and Stops")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
