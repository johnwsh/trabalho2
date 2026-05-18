CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -Iinclude -g
TARGET = floresta.exe
OBJ_DIR = obj

SRCS = main.cpp src/utils.cpp src/trees.cpp src/forest.cpp

OBJS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(SRCS)))

vpath %.cpp src

all: $(TARGET)

$(TARGET): $(OBJS)
	@$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Regra para compilar os .o
$(OBJ_DIR)/%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@rm -rf $(OBJ_DIR) $(TARGET)
