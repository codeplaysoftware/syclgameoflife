#ifndef BENCHMARK_MODE
#include <SDL2/SDL.h>
#endif
#include <array>
#include <iostream>
#include <random>
#include <chrono>
#include <sycl/sycl.hpp>

// MAPWIDTH and MAPHEIGHT must be multiples of GROUPSIZE
constexpr const int MAPWIDTH = 160;
constexpr const int MAPHEIGHT = 160;
constexpr const int GROUPSIZE = 16;
constexpr const int DATASIZE = MAPWIDTH * MAPHEIGHT;
constexpr const int SCALE = 12;
constexpr const int DELAY = 25000;
constexpr const int CHANCEOFLIFE = 25;

class LocalKernel;
class NoLocalKernel;

void initMap(std::array<bool, DATASIZE> &map) {
    std::random_device rd;
    for (int i = 0; i < DATASIZE; i++) {
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distr(0, 100);
        int result = distr(gen);
        map[i] = (result < CHANCEOFLIFE);
    }
}

void lifePass_NoLocal(sycl::queue myQueue, std::array<bool, DATASIZE> &map) {
    std::array<bool, DATASIZE> copy = map;

    auto mapBuf = sycl::buffer<bool, 2>(map.data(), sycl::range<2>{MAPWIDTH, MAPHEIGHT});

    auto copyBuf = sycl::buffer<bool, 2>(copy.data(), sycl::range<2>{MAPWIDTH, MAPHEIGHT});
    copyBuf.set_final_data(nullptr);

    myQueue.submit([&](sycl::handler &cgh) {
            auto copyAcc = copyBuf.get_access<sycl::access::mode::read>(cgh);
            auto mapAcc = mapBuf.get_access<sycl::access::mode::write>(cgh);

            auto range = sycl::range<2>{MAPWIDTH, MAPHEIGHT};

            cgh.parallel_for<NoLocalKernel>(range, [=](sycl::id<2> index) {
                int row = index[0];
                int col = index[1];

                int liveNeighbourCount = 0;
                int neighbours[8] = {
                    copyAcc[row - 1][col - 1], copyAcc[row - 1][col],
                    copyAcc[row - 1][col + 1], copyAcc[row][col + 1],
                    copyAcc[row + 1][col + 1], copyAcc[row + 1][col],
                    copyAcc[row + 1][col - 1], copyAcc[row][col - 1]
                };
                for (auto neighbour : neighbours) {
                    liveNeighbourCount += neighbour;
                }

                if ((mapAcc[row][col]) && liveNeighbourCount < 2 || liveNeighbourCount > 3) {
                    mapAcc[row][col] = false;
                } else if (liveNeighbourCount == 3) {
                    mapAcc[row][col] = true;
                }
            });
        }).wait();
}

void lifePass_Local(sycl::queue myQueue, std::array<bool, DATASIZE> &map) {
    std::array<bool, DATASIZE> copy = map;

    auto mapBuf = sycl::buffer<bool, 2>(map.data(), sycl::range<2>{MAPWIDTH, MAPHEIGHT});

    auto copyBuf = sycl::buffer<bool, 2>(copy.data(), sycl::range<2>{MAPWIDTH, MAPHEIGHT});
    copyBuf.set_final_data(nullptr);

    myQueue.submit([&](sycl::handler &cgh) {
        auto copyAcc = copyBuf.get_access<sycl::access::mode::read>(cgh);
        auto mapAcc = mapBuf.get_access<sycl::access::mode::write>(cgh);

        sycl::range<2> groupSize{GROUPSIZE + 2, GROUPSIZE + 2};
        sycl::range<2> numGroups{size_t(((MAPWIDTH - 1) / GROUPSIZE) + 1),
                                    size_t(((MAPWIDTH - 1) / GROUPSIZE) + 1)};

        cgh.parallel_for_work_group<LocalKernel>(numGroups, groupSize, [=](sycl::group<2> group) {
            bool localMemory[GROUPSIZE + 2][GROUPSIZE + 2] = {{0}};

            group.parallel_for_work_item([&](sycl::h_item<2> item) {
                int localRow = item.get_local_id(0);
                int localCol = item.get_local_id(1);

                sycl::id<2> groupId = group.get_group_id();

                int globalRow = (groupId[0] * GROUPSIZE) + localRow;
                int globalCol = (groupId[1] * GROUPSIZE) + localCol;

                if (globalCol > MAPWIDTH || globalRow > MAPHEIGHT) {
                    localMemory[localRow][localCol] = false;
                } else {
                    localMemory[localRow][localCol] = copyAcc[globalRow - 1][globalCol - 1];
                }
            });

            group.parallel_for_work_item([&](sycl::h_item<2> item) {
                int localRow = item.get_local_id(0);
                int localCol = item.get_local_id(1);

                sycl::id<2> groupId = group.get_group_id();

                int globalRow = (groupId[0] * GROUPSIZE) + localRow;
                int globalCol = (groupId[1] * GROUPSIZE) + localCol;

                if (localRow > 0 && localRow < (GROUPSIZE + 1) && localCol > 0 && localCol < (GROUPSIZE + 1)) {
                    int liveNeighbourCount = 0;
                    int neighbours[8] = {
                        localMemory[localRow - 1][localCol - 1],
                        localMemory[localRow - 1][localCol],
                        localMemory[localRow - 1][localCol + 1],
                        localMemory[localRow][localCol + 1],
                        localMemory[localRow + 1][localCol + 1],
                        localMemory[localRow + 1][localCol],
                        localMemory[localRow + 1][localCol - 1],
                        localMemory[localRow][localCol - 1]
                    };
                    for (auto neighbour : neighbours) {
                        liveNeighbourCount += neighbour;
                    }

                    if ((localMemory[localRow][localCol]) && liveNeighbourCount < 2 || liveNeighbourCount > 3) {
                        mapAcc[globalRow - 1][globalCol - 1] = false;
                    } else if (liveNeighbourCount == 3) {
                        mapAcc[globalRow - 1][globalCol - 1] = true;
                    }
                }
            });
        });
    }).wait();
}

#ifndef BENCHMARK_MODE
void init(SDL_Window *&win, SDL_Renderer *&render) {
    SDL_Init(SDL_INIT_VIDEO);
    win = SDL_CreateWindow("SYCL Game of Life", SDL_WINDOWPOS_UNDEFINED,
                           SDL_WINDOWPOS_UNDEFINED, MAPWIDTH * SCALE,
                           MAPHEIGHT * SCALE, SDL_WINDOW_SHOWN);
    render = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
}
#endif

#ifndef BENCHMARK_MODE
void draw(SDL_Renderer *&render, std::array<bool, DATASIZE> map) {
    SDL_SetRenderDrawColor(render, 255, 255, 255, 255);
    SDL_RenderClear(render);

    SDL_SetRenderDrawColor(render, 0, 0, 0, 255);
    SDL_Rect r;
    for (int i = 0; i < map.size(); i++) {
        if (map[i]) {
            r.x = i % MAPWIDTH * SCALE;
            r.y = i / MAPWIDTH * SCALE;
            r.w = SCALE;
            r.h = SCALE;
            SDL_RenderFillRect(render, &r);
        }
    }

    SDL_SetRenderDrawColor(render, 211, 211, 211, 255);
    for (int i = 0; i < MAPWIDTH; i++) {
        SDL_RenderDrawLine(render, i * SCALE, 0, i * SCALE, MAPHEIGHT * SCALE);
    }
    for (int i = 0; i < MAPHEIGHT; i++) {
        SDL_RenderDrawLine(render, 0, i * SCALE, MAPWIDTH * SCALE, i * SCALE);
    }

    SDL_SetRenderDrawColor(render, 255, 0, 0, 255);
    for (int i = 1; i < (MAPWIDTH / GROUPSIZE); i++) {
        SDL_RenderDrawLine(render, i * GROUPSIZE * SCALE, 0,
                           i * GROUPSIZE * SCALE, MAPHEIGHT * SCALE);
        SDL_RenderDrawLine(render, i * GROUPSIZE * SCALE + 1, 0,
                           i * GROUPSIZE * SCALE + 1, MAPHEIGHT * SCALE);
    }
    for (int i = 1; i < (MAPHEIGHT / GROUPSIZE); i++) {
        SDL_RenderDrawLine(render, 0, i * GROUPSIZE * SCALE, MAPWIDTH * SCALE,
                           i * GROUPSIZE * SCALE);
        SDL_RenderDrawLine(render, 0, i * GROUPSIZE * SCALE + 1,
                           MAPWIDTH * SCALE, i * GROUPSIZE * SCALE + 1);
    }

    SDL_RenderPresent(render);
}
#endif

#ifndef BENCHMARK_MODE
void close(SDL_Window *win, SDL_Renderer *render) {
    SDL_DestroyRenderer(render);
    SDL_DestroyWindow(win);
    SDL_Quit();
}
#endif

int main() {
    #ifndef BENCHMARK_MODE
        SDL_Window *win = NULL;
        SDL_Renderer *render = NULL;
        init(win, render);
    #endif

    sycl::queue myQueue{sycl::gpu_selector()};

    std::array<bool, DATASIZE> map = {};

    sycl::device dev = myQueue.get_device();
    std::cout << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;

    initMap(map);

    #ifndef BENCHMARK_MODE
        draw(render, map);

        int timeCounter = 0;
        bool isQuit = false;
        SDL_Event event;

        while (!isQuit) {
            if (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    isQuit = true;
                }
            }

            if (timeCounter >= DELAY) {
                timeCounter = 0;
                lifePass_Local(myQueue, map);
                draw(render, map);
            } else {
                timeCounter++;
            }
        }

        close(win, render);
    #else
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 500; i++) {
            lifePass_Local(myQueue, map);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << duration.count() << std::endl;
    #endif
    return 0;
}
