/*
 * Vulkan Windowed Program
 *
 * Copyright (C) 2016 Valve Corporation
 * Copyright (C) 2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Enable the WSI extensions

// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED

#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <vulkan/vulkan.hpp>

#include <iostream>
#include <vector>

#include "VulkanDemo.h"

int main()
{
	VulkanDemo demo;
	bool ok = demo.initInstance(800, 800, "Vulkan demo By Amoe");
	if(ok)
		demo.prepare();

    bool stillRunning = true;
	while (stillRunning) {
		demo.run();
		SDL_Event event;
		while (SDL_PollEvent(&event)) {

			switch (event.type) {

			case SDL_QUIT:
				stillRunning = false;
				break;

			default:
				break;
			}
		}

		SDL_Delay(10);
	}
	demo.stop();

    return 0;
}
