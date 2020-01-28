
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// CPU – вариант. Вычисление ускорения
void Acceleration_CPU(float *X, float *Y, float *AX, float *AY,
	int nt, int N, int id)
{
	float ax = 0.f; float ay = 0.f; float xx, yy, rr; int sh = (nt - 1) * N;
	for (int j = 0; j < N; j++) // цикл по частицам
	{
		if (j != id) // поверка самодействия
		{
			xx = X[j + sh] - X[id + sh]; yy = Y[j + sh] - Y[id + sh];
			rr = sqrtf(xx * xx + yy * yy);
			if (rr > 0.01f) // минимальное расстояние 0.01 м
			{
				rr = 10.f / (rr * rr * rr); ax += xx * rr; ay += yy * rr;
			} // if rr
		} // if id
	} // for
	AX[id] = ax;
}

// CPU-вариант. Пересчет координат
void Position_CPU(float *X, float *Y, float *VX,
	float *VY, float *AX, float *AY,
	float tau, int nt, int Np, int id)
{
	int sh = (nt - 1) * Np;
	X[id + nt * Np] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
	Y[id + nt * Np] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;
	VX[id] += AX[id] * tau;
	VY[id] += AY[id] * tau;
}

// GPU-вариант. Расчет ускорения
__global__ void Acceleration_GPU(float *X, float *Y, float *AX, float *AY, int nt, int N)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float ax = 0.f; float ay = 0.f; float xx, yy, rr; int sh = (nt - 1) * N;
	for (int j = 0; j < N; j++) // цикл по частицам
	{
		if (j != id) // проверка самодействия
		{
			xx = X[j + sh] - X[id + sh]; yy = Y[j + sh] - Y[id + sh];
			rr = sqrtf(xx * xx + yy * yy);
			if (rr > 0.01f) // минимальное расстояние 0.01 м
			{
				rr = 10.f / (rr * rr * rr); ax += xx * rr; ay += yy * rr;
			} // if rr
		} // if id
	} // for j
	AX[id] = ax; AY[id] = ay;
}

// GPU-вариант. Пересчет координат
__global__ void Position_GPU(float *X, float *Y, float *VX, float *VY,
	float *AX, float *AY, float tau, int nt, int Np)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int sh = (nt - 1) * Np;
	X[id + nt * Np] = X[id + sh] + VX[id] * tau + AX[id] * tau * tau * 0.5f;
	Y[id + nt * Np] = Y[id + sh] + VY[id] * tau + AY[id] * tau * tau * 0.5f;
	VX[id] += AX[id] * tau;
	VY[id] += AY[id] * tau;
}

__global__ void Acceleration_Shared(float *X, float *Y, float *AX, float *AY, int nt, int N, int N_block)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float ax = 0.f; float ay = 0.f; float xx, yy, rr; int sh = (nt - 1) * N;
	float xxx = X[id + sh]; float yyy = Y[id + sh];
	__shared__ float Xs[256]; __shared__ float Ys[256]; // выделение разделяемой памяти
	for (int i = 0; i < N_block; i++) // основной цикл по блокам
	{
		Xs[threadIdx.x] = X[threadIdx.x + i * blockDim.x + sh]; // копирование из глобальной
		Ys[threadIdx.x] = Y[threadIdx.x + i * blockDim.x + sh]; // в разделяемую память
		__syncthreads(); // синхронизация
		for (int j = 0; j < blockDim.x; j++) // вычислительная часть
		{
			if ((j + i * blockDim.x) != id)
			{
				xx = Xs[j] - xxx; yy = Ys[j] - yyy; rr = sqrtf(xx * xx + yy * yy);
				if (rr > 0.01f) { rr = 10.f / (rr * rr * rr); ax += xx * rr; ay += yy * rr; } //if
			} // if id
		} // for j
		__syncthreads(); // синхронизация
	} // for i
	AX[id] = ax; AY[id] = ay;
}



int main(int argc, char* argv[])
{
	int j = 0, id = 0;
	float timerValueGPU, timerValueCPU;
	cudaEvent_t start, stop; // определение переменных-событий для таймера
	cudaEventCreate(&start); cudaEventCreate(&stop);
	int N = 25600; // число частиц (2-й вариант 20480)
	int NT = 10; // число шагов по времени (для анимации - 800)
	float tau = 0.001f; // шаг по времени 0.001 с
						// создание массивов на host
	float *hX, *hY, *hVX, *hVY, *hAX, *hAY;
	unsigned int mem_size = sizeof(float) * N;
	unsigned int mem_size_big = sizeof(float) * NT * N;
	hX = (float*)malloc(mem_size_big); hY = (float*)malloc(mem_size_big);
	hVX = (float*)malloc(mem_size); hVY = (float*)malloc(mem_size);
	hAX = (float*)malloc(mem_size); hAY = (float*)malloc(mem_size);

	// задание начальных условий на host
	float vv, phi;
	for (j = 0; j < N; j++)
	{
		phi = (float)rand();
		hX[j] = rand() * cosf(phi) * 1.e-4f; hY[j] = rand() * sinf(phi) * 1.e-4f;
		vv = (hX[j] * hX[j] + hY[j] * hY[j]) * 10.f;
		hVX[j] = -vv * sinf(phi); hVY[j] = vv * cosf(phi);
	}
	// создание на device массивов
	float *dX, *dY, *dVX, *dVY, *dAX, *dAY;
	cudaMalloc((void**)&dX, mem_size_big);
	cudaMalloc((void**)&dY, mem_size_big);
	cudaMalloc((void**)&dVX, mem_size); cudaMalloc((void**)&dVY, mem_size);
	cudaMalloc((void**)&dAX, mem_size); cudaMalloc((void**)&dAY, mem_size);
	// задание сетки нитей и блоков
	int N_thread = 256; int N_block = N / N_thread;

	// -----------------GPU-вариант--------------------------
	cudaEventRecord(start, 0);
	// копирование данных на device
	cudaMemcpy(dX, hX, mem_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dY, hY, mem_size_big, cudaMemcpyHostToDevice);
	cudaMemcpy(dVX, hVX, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dVY, hVY, mem_size, cudaMemcpyHostToDevice);
	for (j = 1; j < NT; j++)
	{
		// расчет ускорения
		Acceleration_Shared <<< N_block, N_thread >>> (dX, dY, dAX, dAY, j, N, N_block);
		// пересчет координат
		Position_GPU <<< N_block, N_thread >>> (dX, dY, dVX, dVY, dAX, dAY, tau, j, N);
	}

	// копирование траекторий с device на host
	cudaMemcpy(hX, dX, mem_size_big, cudaMemcpyDeviceToHost);
	cudaMemcpy(hY, dY, mem_size_big, cudaMemcpyDeviceToHost);
	// определение времени выполнения GPU-варианта
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time %f msec\n", timerValueGPU);
	//{...} // сохранение траекторий в файл, GPU-вариант

		  //------------------CPU-вариант------------------------
	//cudaEventRecord(start, 0);
	double startCPU = clock();
	for (j = 1; j < NT; j++)
	{
		for (id = 0; id < N; id++)
		{
			Acceleration_CPU(hX, hY, hAX, hAY, j, N, id);
			Position_CPU(hX, hY, hVX, hVY, hAX, hAY, tau, j, N, id);
		}
	}
	double durationCPU = (clock() - startCPU);
	printf("\n CPU calculation time %f msec\n", durationCPU);
	printf("\n Rate %f x\n", durationCPU / timerValueGPU);
	// определение времени выполнения CPU-варианта
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&timerValueCPU, start, stop);
	//printf("\n CPU calculation time %f msec\n", timerValueCPU );
	//printf("\n Rate %f x\n", timerValueCPU / timerValueGPU);
	//{...} // сохранение траекторий, CPU-вариант

	// освобождение памяти
	free(hX); free(hY); free(hVX); free(hVY); free(hAX); free(hAY);
	cudaFree(dX); cudaFree(dY); cudaFree(dVX); cudaFree(dVY);
	// уничтожение переменных-событий
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}


