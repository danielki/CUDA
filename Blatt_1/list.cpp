#include "list.h"
#include <stdlib.h>
#include <stdio.h>

const unsigned int memory_size_le = sizeof(listElement);	
const unsigned int memory_size_l  = sizeof(listElement);

list* createList()
    {
    list *l = (list*)malloc(memory_size_l);
	l->firstElement = NULL;
	return l;
    }

void addBegin(list *l, int value)
	{
	if (l->firstElement == NULL)
	   {
	   listElement *le = (listElement*)malloc(memory_size_le);
	   le->value = value;
	   le->next = NULL;
	   l->firstElement = le;
	   return;
	   }
	listElement *le = (listElement*)malloc(memory_size_le);
	le->value = value;
	le->next = l->firstElement;
	l->firstElement = le;
	}
listElement* goToEnd(list *l)
	{
	listElement *le= l->firstElement;
	while ( le->next != NULL )
		{
		le = le->next;
		}
	return le;	
	}
	
listElement* goToPosition(list *l, int position)
	{
	int currentPosition = 1;
	listElement *le = l->firstElement;
	if ( position < 1 )
		{
		return le;
		}
	while ( le->next != NULL && currentPosition < position)
		{
		le = le->next;
		currentPosition++;
		}
	return le;	
	}
	
void addEnd(list *l, int value)
	{
	if (l->firstElement == NULL)
	   {
	   listElement *le = (listElement*)malloc(memory_size_le);
	   le->value = value;
	   le->next = NULL;
	   l->firstElement = le;
	   return;
	   }
	listElement *la= goToEnd(l);
	listElement *le = (listElement*)malloc(memory_size_le);
	le->value = value;
	le->next = NULL;
	la->next = le;
	}
	
void addInPosition(list *l, int position, int value)
	{
	if (l->firstElement == NULL)
	   {
	   listElement *le = (listElement*)malloc(memory_size_le);
	   le->value = value;
	   le->next = NULL;
	   l->firstElement = le;
	   return;
	   }
	listElement *le = goToPosition(l, position-1);
	listElement *le_new = (listElement*)malloc(memory_size_le);
	le_new->value = value;
	le_new->next = le->next;
	le->next = le_new;
	}
	
void removeElementAtPosition(list *l, int position)
	{
	if ( lenght == 0 )
	   {
	   return;
	   }
	if ( lenght == 1 )
	   {
	   l->firstElement = NULL;
	   }
	listElement *le = goToPosition(l, position-1);
	listElement *lefree = le->next;
	le->next = le->next->next;
	free(lefree);
	}
	
int removeElementWithValue(list *l, int value)
	{
	int currentPosition = 1;
	listElement *le = l->firstElement;
	int elementRemoved = 0;
	while(le != NULL)
		{
		if ( le->value == value )
			{
			removeElementAtPosition(l, currentPosition);
			elementRemoved++;
			currentPosition--;
			}
		currentPosition++;	
		le = le->next;	
		}
	return elementRemoved;	
	}
	
void print(list *l)
	{
	listElement *le = l->firstElement;
	printf("\n");	
	while(le != NULL )
		{
		printf("%d ",le->value);
		le = le->next;
		}
	printf("\n");	
	}
	
void printValueAtPosition(list *l, int position)
	{
	listElement *le = l->firstElement;
	printf("\n");	
	int currentPosition = 1;
	while(le != NULL )
		{
		if ( currentPosition == position )
		  {
		  printf("%d ",le->value);
		  }
		le = le->next;
		currentPosition++;
		}
	printf("\n");
	}
	
void reverseOrder(list *l)
	{
	if (lenght(l) == 2 )
	   {
	   listElement *le = l->firstElement;
	   listElement *lememn = le->next; 
	   lememn->next = le;
	   le->next = NULL;
	   l->firstElement = lememn;
	   }
	/* Drehe alle Zeiger um und lasse erstes 
	Element auf das letzte zeigen */   
	if (lenght(l) > 2 )
	   {
	   listElement *le = l->firstElement;
	   listElement *lememn = le->next; // memory next
	   listElement *lememnn = lememn->next; // memory next next
	   le->next = NULL;
	   while ( lememnn->next != NULL )
	     {
	     lememn->next = le;
	     le = lememn;
	     lememn = lememnn;
	     lememnn = lememnn->next;
	     }
	   lememn->next = le;  
	   lememnn->next = lememn;   
	   l->firstElement = lememnn;
	   } 
	}
	
int lenght(list *l)
    {
    listElement *le = l->firstElement;
    if ( le == NULL )
        {
        return 0;
        }
    int elementCounter = 1;
	while(le->next != NULL )
		{
		//printf("%d ",elementCounter, " == ");
		le = le->next;
		elementCounter++;
		}
	return elementCounter;
    }





















