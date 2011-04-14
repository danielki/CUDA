#include <stdio.h>
#include "list.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

int RandomNumber(int max)
    {
	return (rand() % (max + 1 ));
    }

int main(int argc, char ** argv)
{
    srand((unsigned) time(NULL));
    list *l = createList();
    addEnd(l,1);
    print(l);
    int parem=10;
    if ( argc >= 2 )
        {
        parem = atoi(argv[1]);
        }
    for ( int x=1; x<=parem; x++ )
    {
    addEnd(l,RandomNumber(9999));
    }
    print(l);
    addBegin(l,4);
    addEnd(l,2);
    print(l);
    printValueAtPosition(l,(lenght(l)/2));
    addInPosition(l, ((lenght(l)/2) + 1) , 300);
    print(l);
    int removedElement = removeElementWithValue(l, 300);
    printf("\nEntfernte Elemente:%d ",removedElement);
    print(l);
    addInPosition(l, ((lenght(l)/2) + 1) , 300);
    addInPosition(l, ((lenght(l)/2) + 1) , 300);
    print(l);
    removedElement = removeElementWithValue(l, 300);
    printf("\nEntfernte Elemente:%d ",removedElement);
    print(l);
    removeElementAtPosition(l, 2);
    print(l);
    reverseOrder(l);
    print(l);
	return 0;
}
