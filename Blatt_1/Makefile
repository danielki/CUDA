CXX = g++ -g -O2
LNK = g++
OBJ = obj/main.o obj/list.o


obj/%.o: %.cpp
	$(CXX) -c $< -o $@

all:
	make test

test: $(OBJ)
	$(LNK) -o bin/program $^

clean:
	rm -f obj/*.o

start:
	./bin/program 5
