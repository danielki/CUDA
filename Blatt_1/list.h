struct listElement
    {
    int value;
    struct listElement *next;
    };

struct list
    {
    struct listElement *firstElement;
    };
list* createList();
void addBegin(list* l, int value);
listElement* goToEnd(list* l);
listElement* goToPosition(list* l, int position);
void addEnd(list* l, int value);
void addInPosition(list* l, int position, int value);
void removeElementAtPosition(list *l, int position);
int removeElementWithValue(list *l, int value);
void print(list *l);
void printValueAtPosition(list *l, int position);
void reverseOrder(list* l);
int lenght(list *l);
