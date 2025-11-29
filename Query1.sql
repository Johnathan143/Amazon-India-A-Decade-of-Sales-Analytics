ALTER TABLE amazon_sales
ADD COLUMN order_year INT ;

UPDATE amazon_sales
SET order_year = YEAR(order_date);