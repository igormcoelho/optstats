all:
	g++ -g -Wfatal-errors --std=c++20 -Ithirdparty/eigen-3.4.0/ -Iinclude/ sometests.cpp -o app_stats
