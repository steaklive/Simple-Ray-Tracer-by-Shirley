// Simple Ray Tracer.cpp from P.Shirley's books series.
// with multithreading and ImageMagick support.

#include "stdafx.h"
#include "float.h"

#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <fstream>  
#include <random>
#include <thread>
#include <mutex>

#include "sphere.h"
#include "moving_sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "bvh_node.h"
#include "texture.h"
#include "aarect.h"
#include "box.h"

#include "Magick++.h"
using namespace Magick;

std::random_device rnd_device_main;
std::mt19937 engine_generator_main(rnd_device_main());
std::uniform_real_distribution<> rnd_dist_main(0, 1);

std::mutex mtx;
const int NUM_THREADS = 4;

const int WIDTH = 960; //Width of an image
const int HEIGHT = 540; //Height of an image
const int NUM_SAMPLES = 25; //AA samples

unsigned char* pBuffer = new unsigned char[WIDTH*HEIGHT*3];

vec3 color(const ray& r, hitable *world, int depth)
{
	hit_record rec;
	if (world->hit(r, 0.001, FLT_MAX, rec)) {
		ray scattered;
		vec3 attenuation;
		vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return emitted + attenuation * color(scattered, world, depth + 1);
		else
			return emitted;
	}
	else
		return vec3(0, 0, 0);
}

hitable *cornell_box() 
{
	hitable **list = new hitable*[8];
	int i = 0;
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *purple = new lambertian(new constant_texture(vec3(0.65, 0.16, 0.93)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));

	list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
	list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
	list[i++] = new xz_rect(213, 343, 227, 332, 554, light);
	list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
	list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
	list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));

	list[i++] = new translate(new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), purple), 15), vec3(265, 0, 295));
	list[i++] = new sphere(vec3(180, 80, 210), 80, new metal(vec3(0.2, 0.90, 0.8), 0.15));
	list[i++] = new sphere(vec3(285, 30, 180), 30, new metal(vec3(0.98, 0.55, 0.1), 0.55));


	return new hitable_list(list, i);
	//return new bvh_node(list, i, 0.0, 1.0);
}

[[deprecated("Use WriteToPNG() for ImageMagick conversion instead!")]]
void WriteToPPM(unsigned char* buffer, int width, int height, const char* fileName)
{
	std::ofstream outfile;
		
	outfile.open(fileName);

	outfile << "P3\n" << width << " " << height << "\n255\n";

	int length = width * height * 3;

	for (int i = 0; i < length; i += 3)
	{
		outfile << buffer[i] << " " << buffer[i + 1] << " " << buffer[i + 2] << "\n";
	}

	outfile.close();
}

void WriteToPNG(unsigned char* buffer, int width, int height, const char* filename)
{
	Image image;
	image.read(width, height, "RGB", CharPixel, buffer);
	image.write(filename);
}

void calculateTracing(camera& cam, hitable* world, float heightBegin, float heightEnd)
{
	float ny = heightEnd;
	for (int j = heightBegin; j < ny; j++)
	{
		for (int i = 0; i < WIDTH; i++)
		{
			vec3 col(0, 0, 0);
			
			for (int s = 0; s < NUM_SAMPLES; s++) {
				float u = float(i + rnd_dist_main(engine_generator_main)) / float(WIDTH);
				float v = float(HEIGHT - j + rnd_dist_main(engine_generator_main)) / float(HEIGHT);

				ray r = cam.get_ray(u, v);
				vec3 p = r.point_at_parameter(2.0);
				col += color(r, world, 0);
			}

			col /= float(NUM_SAMPLES);
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

			pBuffer[ (j)*WIDTH*3 + (3*i)   ] = int(255.9f*vec3::ToFloat(col[0]));
			pBuffer[ (j)*WIDTH*3 + (3*i+1) ] = int(255.9f*vec3::ToFloat(col[1]));
			pBuffer[ (j)*WIDTH*3 + (3*i+2) ] = int(255.9f*vec3::ToFloat(col[2]));
			
		}
	}
}

int main(int argc, char **argv)
{
	auto startTime = std::chrono::system_clock::now();
	
	InitializeMagick(*argv);
	
	hitable* world = cornell_box();
	camera cam(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0, 1, 0), 40.0, float(WIDTH) / float(HEIGHT), 0.0, 10.0, 0.0, 1.0);


	//*******************RAY TRACING WITH MULTITHREADING******************//
	std::vector<std::thread> threads;
	threads.reserve(NUM_THREADS);

	//vertical offset for threads
	float verticalOffset = HEIGHT / NUM_THREADS;

	for (size_t i = 0; i < NUM_THREADS; i++)
	{
		threads.push_back(std::thread(calculateTracing, std::ref(cam), std::ref(world), verticalOffset * i, verticalOffset * i + verticalOffset));
	}
	for (auto& t : threads) t.join();
	//********************************************************************//


	//*********************GENERATING OUTPUT IMAGE************************//
	
	WriteToPNG(pBuffer, WIDTH, HEIGHT, "Result.png");
	//WriteToPPM(pBuffer, WIDTH, HEIGHT, "Image.ppm");

	//********************************************************************//



	std::chrono::duration<double> durationTime = std::chrono::system_clock::now() - startTime;
	std::cout << "Fininshed ray tracing! Total time: " << durationTime.count() << " seconds" << std::endl;

	system("pause");
    return 0;
}

