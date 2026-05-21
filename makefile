CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -g
OBJ_DIR = obj

TARGET_REDE = main_rede.exe
TARGET_FOREST = main_forest.exe

CORE_SRCS = src/utils.cpp src/trees.cpp src/forest.cpp src/neuron.cpp src/layer.cpp src/network.cpp 
CORE_OBJS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(CORE_SRCS)))

vpath %.cpp src .

all: main_rede main_forest

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

main_rede: $(OBJ_DIR) $(TARGET_REDE)
	@echo "[+] Rede Neural compilada com sucesso: $(TARGET_REDE)"

main_forest: $(OBJ_DIR) $(TARGET_FOREST)
	@echo "[+] Random Forest compilada com sucesso: $(TARGET_FOREST)"

$(TARGET_REDE): $(CORE_OBJS) $(OBJ_DIR)/main_rede.o
	@$(CXX) $(CXXFLAGS) -o $@ $^

$(TARGET_FOREST): $(CORE_OBJS) $(OBJ_DIR)/main_forest.o
	@$(CXX) $(CXXFLAGS) -o $@ $^


$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	@echo "Compilando $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Limpando o projeto..."
	@rm -rf $(OBJ_DIR) $(TARGET_REDE) $(TARGET_FOREST)