# Kitchen Manager

> Version 0.1.0

Manage your kitchen inventory

## Path Table

| Method | Path | Description |
| --- | --- | --- |
| GET | [/ingredients/](#getingredients) | Get All Ingredient |
| POST | [/ingredients/](#postingredients) | Create Ingredent |
| GET | [/ingredients/{id}](#getingredientsid) | Get Ingredient |
| PATCH | [/ingredients/{id}](#patchingredientsid) | Update Ingredient |
| DELETE | [/ingredients/{id}](#deleteingredientsid) | Delete Ingredient |
| GET | [/chat/](#getchat) | Chat |

## Reference Table

| Name | Path | Description |
| --- | --- | --- |
| HTTPValidationError | [#/components/schemas/HTTPValidationError](#componentsschemashttpvalidationerror) |  |
| IngredientCreate | [#/components/schemas/IngredientCreate](#componentsschemasingredientcreate) |  |
| IngredientUpdate | [#/components/schemas/IngredientUpdate](#componentsschemasingredientupdate) |  |
| ResponseMultiple_Ingredient_ | [#/components/schemas/ResponseMultiple_Ingredient_](#componentsschemasresponsemultiple_ingredient_) |  |
| ResponseSingle_Ingredient_ | [#/components/schemas/ResponseSingle_Ingredient_](#componentsschemasresponsesingle_ingredient_) |  |
| ResponseSingle_str_ | [#/components/schemas/ResponseSingle_str_](#componentsschemasresponsesingle_str_) |  |
| Unit | [#/components/schemas/Unit](#componentsschemasunit) |  |
| ValidationError | [#/components/schemas/ValidationError](#componentsschemasvalidationerror) |  |

## Path Details

***

### [GET]/ingredients/

- Summary  
Get All Ingredient

- Description  
Get all ingredients

#### Responses

- 200 Successful Response

`application/json`

```ts
{
[]
  message: string
  success: boolean
}
```

- 404 Not Found

***

### [POST]/ingredients/

- Summary  
Create Ingredent

- Description  
Create a new ingredient

#### RequestBody

- application/json

```ts
{
  name: string
  quantity: integer
  unit: enum[kg, g, l, ml, cup, tbsp, piece]
}
```

#### Responses

- 200 Successful Response

`application/json`

```ts
{
  data: Partial() & Partial(null)
  message: string
  success: boolean
}
```

- 404 Not Found

- 422 Validation Error

`application/json`

```ts
{
  detail: {
    loc?: Partial(string) & Partial(integer)[]
    msg: string
    type: string
  }[]
}
```

***

### [GET]/ingredients/{id}

- Summary  
Get Ingredient

- Description  
get a particular ingredient

#### Responses

- 200 Successful Response

`application/json`

```ts
{
  data: Partial() & Partial(null)
  message: string
  success: boolean
}
```

- 404 Not Found

- 422 Validation Error

`application/json`

```ts
{
  detail: {
    loc?: Partial(string) & Partial(integer)[]
    msg: string
    type: string
  }[]
}
```

***

### [PATCH]/ingredients/{id}

- Summary  
Update Ingredient

- Description  
Update a particular ingredient

#### RequestBody

- application/json

```ts
{
  name?: Partial(string) & Partial(null)
  quantity?: Partial(integer) & Partial(null)
  unit?: Partial(#/components/schemas/Unit) & Partial(null)
}
```

#### Responses

- 200 Successful Response

`application/json`

```ts
{
  data: Partial() & Partial(null)
  message: string
  success: boolean
}
```

- 404 Not Found

- 422 Validation Error

`application/json`

```ts
{
  detail: {
    loc?: Partial(string) & Partial(integer)[]
    msg: string
    type: string
  }[]
}
```

***

### [DELETE]/ingredients/{id}

- Summary  
Delete Ingredient

- Description  
Delete a particular ingredient

#### Responses

- 200 Successful Response

`application/json`

```ts
{
  data: Partial() & Partial(null)
  message: string
  success: boolean
}
```

- 404 Not Found

- 422 Validation Error

`application/json`

```ts
{
  detail: {
    loc?: Partial(string) & Partial(integer)[]
    msg: string
    type: string
  }[]
}
```

***

### [GET]/chat/

- Summary  
Chat

- Description  
Query the Chatbot about the recipe  
`query` str: The query to ask the chatbot like "I want to make a cake"

#### Parameters(Query)

```ts
query: string
```

#### Responses

- 200 Successful Response

`application/json`

```ts
{
  data: Partial() & Partial(null)
  message: string
  success: boolean
}
```

- 404 Not Found

- 422 Validation Error

`application/json`

```ts
{
  detail: {
    loc?: Partial(string) & Partial(integer)[]
    msg: string
    type: string
  }[]
}
```

## References

### #/components/schemas/HTTPValidationError

```ts
{
  detail: {
    loc?: Partial(string) & Partial(integer)[]
    msg: string
    type: string
  }[]
}
```

### #/components/schemas/IngredientCreate

```ts
{
  name: string
  quantity: integer
  unit: enum[kg, g, l, ml, cup, tbsp, piece]
}
```

### #/components/schemas/IngredientUpdate

```ts
{
  name?: Partial(string) & Partial(null)
  quantity?: Partial(integer) & Partial(null)
  unit?: Partial(#/components/schemas/Unit) & Partial(null)
}
```

### #/components/schemas/ResponseMultiple_Ingredient_

```ts
{
[]
  message: string
  success: boolean
}
```

### #/components/schemas/ResponseSingle_Ingredient_

```ts
{
  data: Partial() & Partial(null)
  message: string
  success: boolean
}
```

### #/components/schemas/ResponseSingle_str_

```ts
{
  data: Partial() & Partial(null)
  message: string
  success: boolean
}
```

### #/components/schemas/Unit

```ts
{
  "type": "string",
  "enum": [
    "kg",
    "g",
    "l",
    "ml",
    "cup",
    "tbsp",
    "piece"
  ],
  "title": "Unit"
}
```

### #/components/schemas/ValidationError

```ts
{
  loc?: Partial(string) & Partial(integer)[]
  msg: string
  type: string
}
```
